#!/usr/bin/env python3
"""
3D Hand Pose Estimation - Inference Script (Simplified for MediaPipe 0.10+)
===========================================================================

Compatible with MediaPipe 0.10.30+ which uses the new Tasks API.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import sys
import os
from pathlib import Path
import urllib.request


class HandPoseEstimator:
    """3D Hand Pose Estimator using MediaPipe Hands (Tasks API)"""
    
    JOINT_NAMES = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]
    
    PARENT_INDICES = [
        -1,  # WRIST (root)
        0, 1, 2, 3,  # THUMB chain
        0, 5, 6, 7,  # INDEX chain
        0, 9, 10, 11,  # MIDDLE chain
        0, 13, 14, 15,  # RING chain
        0, 17, 18, 19   # PINKY chain
    ]
    
    def __init__(self, min_detection_confidence=0.5):
        """Initialize MediaPipe Hands with Tasks API"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download model if needed
            model_path = self._get_model_path()
            
            # Create hand landmarker
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.mp = mp
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe: {e}")
    
    def _get_model_path(self):
        """Download or get path to hand landmarker model"""
        model_dir = Path.home() / '.mediapipe' / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'hand_landmarker.task'
        
        if not model_path.exists():
            print("Downloading MediaPipe hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✓ Model downloaded to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        
        return str(model_path)
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_rgb
    
    def detect_hand_landmarks(self, image_rgb):
        """Detect 3D hand landmarks"""
        try:
            # Create MediaPipe Image
            mp_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=image_rgb
            )
            
            # Detect
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.hand_world_landmarks:
                raise ValueError("No hand detected in the image")
            
            # Extract 3D world coordinates
            world_landmarks = detection_result.hand_world_landmarks[0]
            landmarks_3d = np.array([
                [lm.x, lm.y, lm.z] for lm in world_landmarks
            ])
            
            return landmarks_3d, detection_result
            
        except Exception as e:
            if "No hand detected" in str(e):
                raise
            raise RuntimeError(f"Detection failed: {e}")
    
    def compute_rotation_from_vectors(self, v1, v2):
        """Compute rotation matrix that aligns v1 to v2"""
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            return np.eye(3)
        
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        
        rotation = R.from_rotvec(angle * axis)
        return rotation.as_matrix()
    
    def compute_joint_rotations(self, landmarks_3d):
        """Compute joint rotations from 3D landmarks"""
        joint_rotations = {}
        
        # Rest pose directions
        rest_pose_directions = {
            1: np.array([1.0, 0.2, 0.0]), 2: np.array([1.0, 0.1, 0.0]),
            3: np.array([1.0, 0.0, 0.0]), 4: np.array([1.0, 0.0, 0.0]),
            5: np.array([0.0, 1.0, 0.0]), 6: np.array([0.0, 1.0, 0.0]),
            7: np.array([0.0, 1.0, 0.0]), 8: np.array([0.0, 1.0, 0.0]),
            9: np.array([0.0, 1.0, 0.0]), 10: np.array([0.0, 1.0, 0.0]),
            11: np.array([0.0, 1.0, 0.0]), 12: np.array([0.0, 1.0, 0.0]),
            13: np.array([0.0, 1.0, 0.0]), 14: np.array([0.0, 1.0, 0.0]),
            15: np.array([0.0, 1.0, 0.0]), 16: np.array([0.0, 1.0, 0.0]),
            17: np.array([0.0, 1.0, 0.0]), 18: np.array([0.0, 1.0, 0.0]),
            19: np.array([0.0, 1.0, 0.0]), 20: np.array([0.0, 1.0, 0.0])
        }
        
        for i, joint_name in enumerate(self.JOINT_NAMES):
            parent_idx = self.PARENT_INDICES[i]
            
            if parent_idx == -1:
                # Wrist orientation from palm normal
                index_mcp = landmarks_3d[5] - landmarks_3d[0]
                pinky_mcp = landmarks_3d[17] - landmarks_3d[0]
                palm_normal = np.cross(index_mcp, pinky_mcp)
                rest_up = np.array([0.0, 0.0, 1.0])
                rotation_matrix = self.compute_rotation_from_vectors(rest_up, palm_normal)
            else:
                # Compute bone vector
                bone_vector = landmarks_3d[i] - landmarks_3d[parent_idx]
                rest_direction = rest_pose_directions.get(i, np.array([0.0, 1.0, 0.0]))
                rotation_matrix = self.compute_rotation_from_vectors(rest_direction, bone_vector)
            
            # Convert to multiple formats
            scipy_rotation = R.from_matrix(rotation_matrix)
            euler_xyz = scipy_rotation.as_euler('xyz', degrees=True)
            quat = scipy_rotation.as_quat()  # [x, y, z, w]
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            rotvec = scipy_rotation.as_rotvec()
            angle = np.linalg.norm(rotvec)
            axis = rotvec / (angle + 1e-8) if angle > 1e-6 else np.array([0, 0, 1])
            
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
    
    def visualize_results(self, image, detection_result, image_path):
        """Create visualization with landmarks"""
        if not detection_result.hand_landmarks:
            return
        
        # Draw landmarks
        hand_landmarks = detection_result.hand_landmarks[0]
        h, w = image.shape[:2]
        
        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            start_px = (int(start_point.x * w), int(start_point.y * h))
            end_px = (int(end_point.x * w), int(end_point.y * h))
            cv2.line(image, start_px, end_px, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            px = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(image, px, 5, (0, 0, 255), -1)
        
        # Save
        output_path = str(Path(image_path).with_suffix('')) + '_visualization.jpg'
        cv2.imwrite(output_path, image)
        print(f"✓ Visualization saved to: {output_path}")
    
    def infer(self, image_path, output_json=None, visualize=False):
        """Run full inference pipeline"""
        print(f"Loading image: {image_path}")
        image, image_rgb = self.load_image(image_path)
        
        print("Detecting hand landmarks...")
        landmarks_3d, detection_result = self.detect_hand_landmarks(image_rgb)
        
        print("Computing joint rotations...")
        joint_rotations = self.compute_joint_rotations(landmarks_3d)
        
        results = {
            "image_path": str(image_path),
            "model": "MediaPipe Hands 0.10+ (pretrained)",
            "num_joints": len(joint_rotations),
            "joint_rotations": joint_rotations,
            "coordinate_system": {
                "description": "Right-hand coordinate system, origin at wrist",
                "units": "meters for position, degrees/radians for rotation",
                "euler_convention": "XYZ (roll-pitch-yaw)"
            }
        }
        
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to: {output_json}")
        
        if visualize:
            self.visualize_results(image, detection_result, image_path)
        
        return results
    
    def print_joint_rotations(self, results):
        """Print joint rotations"""
        print("\n" + "="*70)
        print("JOINT ROTATIONS (for robotic/prosthetic hand control)")
        print("="*70)
        
        for joint_name, rotation_data in results["joint_rotations"].items():
            euler = rotation_data["euler_xyz_deg"]
            quat = rotation_data["quaternion_wxyz"]
            print(f"\n{joint_name}:")
            print(f"  Euler XYZ (deg): [{euler[0]:7.2f}, {euler[1]:7.2f}, {euler[2]:7.2f}]")
            print(f"  Quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
        
        print("\n" + "="*70)


def main():
    """Main inference function"""
    if len(sys.argv) < 2:
        print("Usage: python infer_hand_pose_v2.py <image_path> [--output output.json] [--visualize]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_json = None
    visualize = False
    
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_json = sys.argv[idx + 1]
    
    if "--visualize" in sys.argv:
        visualize = True
    
    if output_json is None:
        output_json = str(Path(image_path).with_suffix('')) + '_pose.json'
    
    try:
        estimator = HandPoseEstimator(min_detection_confidence=0.5)
        results = estimator.infer(image_path, output_json, visualize)
        estimator.print_joint_rotations(results)
        print(f"\n✓ Inference complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
