#!/usr/bin/env python3
"""
3D Hand Pose Estimation - Inference Script
===========================================

This script takes a single RGB image of a hand and outputs 3D joint rotation values
suitable for driving robotic/prosthetic hands.

Pipeline:
1. Detect hand landmarks using MediaPipe (pretrained CNN)
2. Estimate 3D joint positions in camera space
3. Compute joint rotations from the kinematic chain
4. Output rotations in Euler angles (XYZ convention)

Joint Hierarchy (21 keypoints):
- WRIST (0)
- THUMB: CMC(1), MCP(2), IP(3), TIP(4)
- INDEX: MCP(5), PIP(6), DIP(7), TIP(8)
- MIDDLE: MCP(9), PIP(10), DIP(11), TIP(12)
- RING: MCP(13), PIP(14), DIP(15), TIP(16)
- PINKY: MCP(17), PIP(18), DIP(19), TIP(20)

Note: MediaPipe provides world coordinates which include depth information
suitable for inferring occluded joints using anatomical priors.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import sys
import os
from pathlib import Path

# Handle different MediaPipe API versions
try:
    import mediapipe as mp
    # Try new API (0.10+)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_NEW_API = True
except (ImportError, AttributeError):
    # Fall back to old API
    import mediapipe as mp
    MEDIAPIPE_NEW_API = False


class HandPoseEstimator:
    """
    3D Hand Pose Estimator using MediaPipe Hands
    
    MediaPipe Hands uses a pretrained ML pipeline:
    - Palm detection model (SSD-based)
    - Hand landmark model (CNN-based)
    - 3D world landmark prediction with anatomical consistency
    """
    
    # Joint names following robotic hand conventions
    JOINT_NAMES = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]
    
    # Kinematic chain: parent joint for each joint
    # -1 means root (wrist)
    PARENT_INDICES = [
        -1,  # WRIST (root)
        0, 1, 2, 3,  # THUMB chain
        0, 5, 6, 7,  # INDEX chain
        0, 9, 10, 11,  # MIDDLE chain
        0, 13, 14, 15,  # RING chain
        0, 17, 18, 19   # PINKY chain
    ]
    
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Hands with pretrained models
        
        Args:
            model_complexity: 0 (lite), 1 (full), or 2 (heavy) - trades speed for accuracy
            min_detection_confidence: Minimum confidence for palm detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        if MEDIAPIPE_NEW_API:
            # Use new MediaPipe Tasks API (0.10+)
            # Download model if needed
            model_path = self._download_hand_landmarker_model()
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.mp_hands = None
        else:
            # Use old MediaPipe Solutions API
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.detector = None
    
    def _download_hand_landmarker_model(self):
        """Download MediaPipe hand landmarker model if not present"""
        import urllib.request
        
        model_dir = Path.home() / '.mediapipe' / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'hand_landmarker.task'
        
        if not model_path.exists():
            print("Downloading MediaPipe hand landmarker model (first time only)...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")
        
        return str(model_path)
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_rgb
    
    def detect_hand_landmarks(self, image_rgb):
        """
        Detect 3D hand landmarks using pretrained MediaPipe model
        
        Returns:
            landmarks_3d: (21, 3) array of 3D world coordinates in meters
                         These are real-world 3D positions with anatomical constraints
        """
        if MEDIAPIPE_NEW_API:
            # Use new API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.hand_world_landmarks:
                raise ValueError("No hand detected in the image")
            
            # Get world landmarks (3D coordinates in meters)
            world_landmarks = detection_result.hand_world_landmarks[0]
            
            # Extract 3D coordinates
            landmarks_3d = np.array([
                [lm.x, lm.y, lm.z] for lm in world_landmarks
            ])
        else:
            # Use old API
            results = self.hands.process(image_rgb)
            
            if not results.multi_hand_landmarks or not results.multi_hand_world_landmarks:
                raise ValueError("No hand detected in the image")
            
            # Get world landmarks (3D coordinates in meters, origin at wrist)
            world_landmarks = results.multi_hand_world_landmarks[0]
            
            # Extract 3D coordinates
            landmarks_3d = np.array([
                [lm.x, lm.y, lm.z] for lm in world_landmarks.landmark
            ])
        
        return landmarks_3d
    
    def compute_rotation_from_vectors(self, v1, v2):
        """
        Compute rotation matrix that aligns v1 to v2
        
        Uses Rodrigues' rotation formula for robust rotation estimation
        """
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Cross product gives rotation axis
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # Vectors are parallel
            return np.eye(3)
        
        axis = axis / axis_norm
        
        # Dot product gives rotation angle
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        
        # Convert to rotation matrix using scipy
        rotation = R.from_rotvec(angle * axis)
        return rotation.as_matrix()
    
    def compute_joint_rotations(self, landmarks_3d):
        """
        Compute joint rotations from 3D landmark positions
        
        Strategy:
        - For each joint, compute the bone vector (from parent to joint)
        - Compute rotation that aligns the rest pose bone to the current bone
        - Extract Euler angles (XYZ convention) for robotic control
        
        Args:
            landmarks_3d: (21, 3) array of 3D world coordinates
            
        Returns:
            joint_rotations: dict mapping joint names to rotation dictionaries
        """
        joint_rotations = {}
        
        # Define rest pose bone directions (neutral hand pose)
        # These are approximate anatomical directions
        rest_pose_directions = {
            # Thumb points outward and slightly up
            1: np.array([1.0, 0.2, 0.0]),
            2: np.array([1.0, 0.1, 0.0]),
            3: np.array([1.0, 0.0, 0.0]),
            4: np.array([1.0, 0.0, 0.0]),
            # Index points forward
            5: np.array([0.0, 1.0, 0.0]),
            6: np.array([0.0, 1.0, 0.0]),
            7: np.array([0.0, 1.0, 0.0]),
            8: np.array([0.0, 1.0, 0.0]),
            # Middle points forward
            9: np.array([0.0, 1.0, 0.0]),
            10: np.array([0.0, 1.0, 0.0]),
            11: np.array([0.0, 1.0, 0.0]),
            12: np.array([0.0, 1.0, 0.0]),
            # Ring points forward
            13: np.array([0.0, 1.0, 0.0]),
            14: np.array([0.0, 1.0, 0.0]),
            15: np.array([0.0, 1.0, 0.0]),
            16: np.array([0.0, 1.0, 0.0]),
            # Pinky points forward
            17: np.array([0.0, 1.0, 0.0]),
            18: np.array([0.0, 1.0, 0.0]),
            19: np.array([0.0, 1.0, 0.0]),
            20: np.array([0.0, 1.0, 0.0])
        }
        
        for i, joint_name in enumerate(self.JOINT_NAMES):
            parent_idx = self.PARENT_INDICES[i]
            
            if parent_idx == -1:
                # Wrist (root joint) - compute orientation from palm normal
                # Use cross product of index and pinky MCP bones
                index_mcp = landmarks_3d[5] - landmarks_3d[0]
                pinky_mcp = landmarks_3d[17] - landmarks_3d[0]
                palm_normal = np.cross(index_mcp, pinky_mcp)
                
                # Compute rotation from standard up direction
                rest_up = np.array([0.0, 0.0, 1.0])
                rotation_matrix = self.compute_rotation_from_vectors(rest_up, palm_normal)
                
            else:
                # Compute bone vector (from parent to current joint)
                bone_vector = landmarks_3d[i] - landmarks_3d[parent_idx]
                
                # Get rest pose direction
                if i in rest_pose_directions:
                    rest_direction = rest_pose_directions[i]
                else:
                    rest_direction = np.array([0.0, 1.0, 0.0])
                
                # Compute rotation from rest pose to current pose
                rotation_matrix = self.compute_rotation_from_vectors(rest_direction, bone_vector)
            
            # Convert rotation matrix to different representations
            scipy_rotation = R.from_matrix(rotation_matrix)
            
            # Euler angles (XYZ convention, in degrees) - most common for robotics
            euler_xyz = scipy_rotation.as_euler('xyz', degrees=True)
            
            # Quaternion (w, x, y, z) - more stable for interpolation
            quat = scipy_rotation.as_quat()  # Returns [x, y, z, w]
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            
            # Axis-angle representation (rotation vector)
            rotvec = scipy_rotation.as_rotvec()
            angle = np.linalg.norm(rotvec)
            axis = rotvec / (angle + 1e-8) if angle > 1e-6 else np.array([0, 0, 1])
            
            # Store all rotation representations
            joint_rotations[joint_name] = {
                "euler_xyz_deg": euler_xyz.tolist(),
                "euler_xyz_rad": (euler_xyz * np.pi / 180).tolist(),
                "quaternion_wxyz": quat_wxyz.tolist(),
                "axis_angle": {
                    "axis": axis.tolist(),
                    "angle_rad": float(angle),
                    "angle_deg": float(angle * 180 / np.pi)
                },
                "rotation_vector": rotvec.tolist(),
                "position_3d_m": landmarks_3d[i].tolist()
            }
        
        return joint_rotations
    
    def infer(self, image_path, output_json=None, visualize=False):
        """
        Run full inference pipeline on a single image
        
        Args:
            image_path: Path to input image
            output_json: Optional path to save results as JSON
            visualize: Whether to save visualization
            
        Returns:
            results: Dictionary containing joint rotations and metadata
        """
        print(f"Loading image: {image_path}")
        image, image_rgb = self.load_image(image_path)
        
        print("Detecting hand landmarks using pretrained MediaPipe model...")
        landmarks_3d = self.detect_hand_landmarks(image_rgb)
        
        print("Computing joint rotations from 3D landmarks...")
        joint_rotations = self.compute_joint_rotations(landmarks_3d)
        
        # Prepare output
        results = {
            "image_path": str(image_path),
            "model": "MediaPipe Hands (pretrained)",
            "num_joints": len(joint_rotations),
            "joint_rotations": joint_rotations,
            "coordinate_system": {
                "description": "Right-hand coordinate system, origin at wrist",
                "units": "meters for position, degrees/radians for rotation",
                "euler_convention": "XYZ (roll-pitch-yaw)"
            }
        }
        
        # Save JSON output
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_json}")
        
        # Visualize (optional)
        if visualize:
            self.visualize_results(image, landmarks_3d, image_path)
        
        return results
    
    def visualize_results(self, image, landmarks_3d, image_path):
        """
        Create visualization with detected landmarks
        """
        if MEDIAPIPE_NEW_API:
            # Use new API - need to re-detect for visualization
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.hand_landmarks:
                # Draw landmarks manually
                from mediapipe import solutions
                from mediapipe.framework.formats import landmark_pb2
                
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in detection_result.hand_landmarks[0]
                ])
                
                solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )
        else:
            # Use old API
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            # Process again to get image landmarks for visualization
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        # Save visualization
        output_path = str(Path(image_path).with_suffix('')) + '_visualization.jpg'
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")
    
    def print_joint_rotations(self, results):
        """Print joint rotations in a readable format"""
        print("\n" + "="*70)
        print("JOINT ROTATIONS (suitable for robotic/prosthetic hand control)")
        print("="*70)
        
        for joint_name, rotation_data in results["joint_rotations"].items():
            euler = rotation_data["euler_xyz_deg"]
            quat = rotation_data["quaternion_wxyz"]
            
            print(f"\n{joint_name}:")
            print(f"  Euler XYZ (deg): [{euler[0]:7.2f}, {euler[1]:7.2f}, {euler[2]:7.2f}]")
            print(f"  Quaternion (w,x,y,z): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
        
        print("\n" + "="*70)


def main():
    """
    Main inference function
    
    Usage:
        python infer_hand_pose.py <image_path> [--output output.json] [--visualize]
    """
    if len(sys.argv) < 2:
        print("Usage: python infer_hand_pose.py <image_path> [--output output.json] [--visualize]")
        print("\nExample:")
        print("  python infer_hand_pose.py hand_image.jpg --output results.json --visualize")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Parse optional arguments
    output_json = None
    visualize = False
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_json = sys.argv[idx + 1]
    
    if "--visualize" in sys.argv:
        visualize = True
    
    # If no output specified, create default
    if output_json is None:
        output_json = str(Path(image_path).with_suffix('')) + '_pose.json'
    
    try:
        # Initialize estimator
        estimator = HandPoseEstimator(
            model_complexity=1,  # Balance between speed and accuracy
            min_detection_confidence=0.5
        )
        
        # Run inference
        results = estimator.infer(
            image_path=image_path,
            output_json=output_json,
            visualize=visualize
        )
        
        # Print results to console
        estimator.print_joint_rotations(results)
        
        print(f"\n✓ Inference complete!")
        print(f"✓ Joint rotations saved to: {output_json}")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
