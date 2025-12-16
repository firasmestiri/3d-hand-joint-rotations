"""
Quick test script to verify the hand pose estimation pipeline

This script can be used to test the installation and basic functionality
without requiring an actual hand image.
"""

import numpy as np
from infer_hand_pose import HandPoseEstimator


def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
        print(f"  MediaPipe version: {mp.__version__}")
        if mp.__version__ != "0.10.9":
            print(f"  Warning: MediaPipe {mp.__version__} detected. Recommended version is 0.10.9 for Windows compatibility.")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from scipy.spatial.transform import Rotation
        print("✓ SciPy imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    return True


def test_estimator_initialization():
    """Test that the HandPoseEstimator can be initialized"""
    print("\nTesting HandPoseEstimator initialization...")
    
    try:
        estimator = HandPoseEstimator(
            model_complexity=1,
            min_detection_confidence=0.5
        )
        print("✓ HandPoseEstimator initialized successfully")
        return True
    except Exception as e:
        print(f"✗ HandPoseEstimator initialization failed: {e}")
        return False


def test_rotation_computation():
    """Test rotation computation functions"""
    print("\nTesting rotation computation...")
    
    try:
        estimator = HandPoseEstimator()
        
        # Test with simple vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        
        rotation_matrix = estimator.compute_rotation_from_vectors(v1, v2)
        
        assert rotation_matrix.shape == (3, 3), "Rotation matrix should be 3x3"
        print("✓ Rotation computation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Rotation computation failed: {e}")
        return False


def test_joint_names():
    """Verify joint naming structure"""
    print("\nTesting joint structure...")
    
    try:
        estimator = HandPoseEstimator()
        
        assert len(estimator.JOINT_NAMES) == 21, "Should have 21 joints"
        assert len(estimator.PARENT_INDICES) == 21, "Should have 21 parent indices"
        
        print(f"✓ Joint structure correct: {len(estimator.JOINT_NAMES)} joints")
        print(f"  Joints: {', '.join(estimator.JOINT_NAMES[:5])}...")
        return True
        
    except Exception as e:
        print(f"✗ Joint structure test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Hand Pose Estimation - Installation Test")
    print("="*60)
    
    tests = [
        test_imports,
        test_estimator_initialization,
        test_rotation_computation,
        test_joint_names
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "="*60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*60)
    
    if all(results):
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Prepare a hand image (.jpg or .png)")
        print("2. Run: python infer_hand_pose_v2.py your_hand_image.jpg --visualize")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
