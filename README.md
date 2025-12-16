# 3D Hand Pose Estimation for Robotic Control

A minimal working prototype that takes a single RGB image of a hand gesture and outputs 3D joint rotation values for all hand joints, including occluded ones. Designed for prosthetic/robotic hand control applications.

> **📌 Quick Start:** Use `infer_hand_pose_v2.py` (Windows-compatible version with MediaPipe 0.10.9)  
> **⚠️ Important:** MediaPipe 0.10.9 is required for Windows compatibility (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md))

## Features

✅ **Pretrained Model**: Uses MediaPipe Hands (Google Research) - no training required  
✅ **Joint Rotations**: Outputs Euler angles, quaternions, and axis-angle representations  
✅ **Occlusion Handling**: Infers occluded joints using anatomical priors  
✅ **Robotic Ready**: Output format suitable for direct robot/prosthetic control  
✅ **Single Image Input**: Works on static images (.jpg, .png)  
✅ **CPU/GPU Compatible**: Runs on both CPU and GPU

## Architecture

**Pipeline:**
1. **Hand Detection**: MediaPipe's palm detection model (SSD-based)
2. **3D Landmark Estimation**: CNN-based model predicts 21 3D keypoints
3. **Anatomical Constraints**: World coordinates maintain hand proportions
4. **Rotation Computation**: Kinematic chain calculates joint rotations
5. **Multi-format Output**: Euler, quaternion, axis-angle for flexibility

**Joint Structure (21 keypoints):**
- Wrist (root)
- Thumb: CMC, MCP, IP, TIP (4 joints)
- Index: MCP, PIP, DIP, TIP (4 joints)
- Middle: MCP, PIP, DIP, TIP (4 joints)
- Ring: MCP, PIP, DIP, TIP (4 joints)
- Pinky: MCP, PIP, DIP, TIP (4 joints)

## Installation

### 1. Prerequisites
- Recommended : Python 3.8+, can work with previous versions
- pip
- Windows, macOS, or Linux

### 2. Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

**Important:** This project uses MediaPipe 0.10.9 due to compatibility issues with newer versions (0.10.30+) on Windows. The requirements.txt file specifies the correct version.

**Core Dependencies:**
- `torch` - Deep learning framework
- `mediapipe` - Pretrained hand detection/tracking models
- `opencv-python` - Image processing
- `scipy` - Rotation conversions
- `numpy` - Numerical operations

### 3. Verify Installation

```powershell
python -c "import mediapipe as mp; import torch; print('Installation successful!')"
```

## Usage

### Basic Usage

```powershell
python infer_hand_pose_v2.py hand_image.jpg
```

This will:
- Detect the hand in `hand_image.jpg`
- Compute 3D joint rotations
- Save results to `hand_image_pose.json`
- Print rotations to console

### Advanced Usage

```powershell
# Specify custom output file
python infer_hand_pose_v2.py hand_image.jpg --output my_results.json

# Generate visualization with landmarks
python infer_hand_pose_v2.py hand_image.jpg --visualize

# Both options together
python infer_hand_pose_v2.py hand_image.jpg --output results.json --visualize
```

## Output Format

The script outputs a JSON file with the following structure:

```json
{
  "image_path": "hand_image.jpg",
  "model": "MediaPipe Hands (pretrained)",
  "num_joints": 21,
  "coordinate_system": {
    "description": "Right-hand coordinate system, origin at wrist",
    "units": "meters for position, degrees/radians for rotation",
    "euler_convention": "XYZ (roll-pitch-yaw)"
  },
  "joint_rotations": {
    "WRIST": {
      "euler_xyz_deg": [10.5, -5.2, 3.8],
      "euler_xyz_rad": [0.183, -0.091, 0.066],
      "quaternion_wxyz": [0.998, 0.045, -0.027, 0.033],
      "axis_angle": {
        "axis": [0.707, -0.408, 0.577],
        "angle_rad": 0.196,
        "angle_deg": 11.23
      },
      "rotation_vector": [0.139, -0.080, 0.113],
      "position_3d_m": [0.0, 0.0, 0.0]
    },
    "THUMB_CMC": { ... },
    "THUMB_MCP": { ... },
    "INDEX_MCP": { ... }
  }
}
```

### Understanding the Output

**For Robotic Control, use one of:**

1. **Euler Angles (XYZ)** - Most common for servo motors
   - `euler_xyz_deg`: Roll, Pitch, Yaw in degrees
   - Range: typically -180° to 180°
   - Apply sequentially: rotate X, then Y, then Z

2. **Quaternions (w, x, y, z)** - Best for smooth interpolation
   - `quaternion_wxyz`: [w, x, y, z] format
   - Normalized: w² + x² + y² + z² = 1
   - No gimbal lock, efficient composition

3. **Axis-Angle** - Intuitive single rotation
   - `axis`: 3D unit vector defining rotation axis
   - `angle_rad`/`angle_deg`: rotation magnitude
   - Direct mapping to revolute joints

## Example Application

### Python Script to Control Robot

```python
import json

# Load pose data
with open('hand_image_pose.json', 'r') as f:
    pose_data = json.load(f)

# Extract rotations for specific joints
thumb_mcp_euler = pose_data['joint_rotations']['THUMB_MCP']['euler_xyz_deg']
index_pip_euler = pose_data['joint_rotations']['INDEX_PIP']['euler_xyz_deg']

# Send to robot controller (pseudo-code)
robot.set_joint_angle('thumb_mcp', thumb_mcp_euler)
robot.set_joint_angle('index_pip', index_pip_euler)
```

## Technical Details

### How Occlusion is Handled

MediaPipe's hand landmark model is trained on diverse hand poses with partial occlusions. The model:
- Uses a CNN with **anatomical priors** learned during training
- Predicts **3D world coordinates** that maintain valid hand proportions
- Enforces **kinematic constraints** (bone lengths, joint limits)
- Provides **confidence scores** for each landmark

Even when fingers are occluded (e.g., behind the palm), the model infers their positions based on:
- Visible wrist/palm orientation
- Typical hand anatomy
- Learned pose manifolds from training data

### Coordinate System

- **Origin**: Wrist joint (landmark 0)
- **Units**: Meters (real-world scale)
- **Axes**: Right-hand coordinate system
  - X: Lateral (thumb side)
  - Y: Longitudinal (fingertips direction)
  - Z: Out of palm

### Model Information

**MediaPipe Hands:**
- **Detection Model**: BlazePalm (SSD MobileNetV2 backbone)
- **Landmark Model**: Custom CNN with 3D coordinate regression
- **Training Data**: 30K+ hand images with 3D annotations
- **Inference Speed**: ~30-50 FPS on modern CPU
- **Accuracy**: Sub-5mm average 3D error on visible landmarks

## Limitations & Considerations

1. **Single Hand**: Currently detects only one hand per image
2. **Visibility**: Best results when palm or back of hand is visible
3. **Extreme Poses**: Very unusual hand poses may have lower accuracy
4. **Scale**: Output is scale-normalized; absolute depth is approximate
5. **Handedness**: Model works for both left and right hands

## Troubleshooting

### "function 'free' not found" or MediaPipe import errors
- **Cause:** MediaPipe 0.10.30+ has Windows compatibility issues
- **Solution:** Ensure you're using MediaPipe 0.10.9 (specified in requirements.txt)
- **Fix:** `pip uninstall mediapipe && pip install mediapipe==0.10.9`

### "module 'mediapipe' has no attribute 'solutions'"
- Use `infer_hand_pose_v2.py` which is compatible with MediaPipe 0.10.9+
- Ensure virtual environment is activated

### "No hand detected in the image"
- Ensure hand occupies significant portion of image (>15% of frame)
- Check lighting conditions (avoid extreme shadows)
- Verify hand is clearly visible (not mostly occluded by objects)

### Low accuracy on specific gestures
- Try adjusting `min_detection_confidence` parameter (default 0.5)
- Ensure reference gestures match expected hand orientation

### Performance issues
- Resize large images before processing
- First run downloads model (~3MB) and may be slower
- Consider GPU acceleration (PyTorch GPU support)

## Extensions & Next Steps

### For Production Use:
1. **Calibration**: Add camera calibration for better 3D accuracy
2. **Temporal Smoothing**: Average rotations across video frames
3. **Joint Limits**: Apply robot-specific joint constraints
4. **Hand Detection**: Add left/right hand classification
5. **Multiple Hands**: Support detection of both hands

### Alternative Models:
- **FrankMocap**: More accurate MANO parameters (requires more setup)
- **Hand4Whole**: Full body + hand pose estimation
- **HaMeR**: Transformer-based hand mesh recovery

## Citation

If using MediaPipe in research:

```
@article{mediapipe2020,
  title={MediaPipe Hands: On-device Real-time Hand Tracking},
  author={Zhang, Fan and Bazarevsky, Valentin and Vakunov, Andrey and Tkachenka, Andrei and Sung, George and Chang, Chuo-Ling and Grundmann, Matthias},
  journal={arXiv preprint arXiv:2006.10214},
  year={2020}
}
```

## License

This implementation is for research purposes. MediaPipe is licensed under Apache 2.0.

## Support

For issues related to:
- **This script**: Check that dependencies are correctly installed
- **MediaPipe**: See [MediaPipe documentation](https://google.github.io/mediapipe/)
- **Hand gestures in images**: Refer to the gesture reference table in your research materials
