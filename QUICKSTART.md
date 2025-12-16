# Quick Start Guide

## 1. Setup (First Time Only)

### Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Test Installation
```powershell
python test_installation.py
```

Expected output: "All tests passed!"

---

## 2. Run Inference on Your Hand Image

### Basic Usage
```powershell
python infer_hand_pose_v2.py your_hand_image.jpg
```

### With Visualization
```powershell
python infer_hand_pose_v2.py your_hand_image.jpg --visualize
```

### Custom Output Path
```powershell
python infer_hand_pose_v2.py your_hand_image.jpg --output my_results.json --visualize
```

**Note:** Use `infer_hand_pose_v2.py` (the working version). MediaPipe 0.10.9 is used for Windows compatibility.

---

## 3. Understanding the Output

The script generates a JSON file with this structure:

```json
{
  "joint_rotations": {
    "WRIST": {
      "euler_xyz_deg": [roll, pitch, yaw],  ← Use this for servo motors
      "quaternion_wxyz": [w, x, y, z],      ← Use this for smooth control
      "axis_angle": {...}                   ← Use this for revolute joints
    },
    "THUMB_MCP": {...},
    "INDEX_PIP": {...}
  }
}
```

---

## 4. Using the Output in Your Robot Code

### Example: Extract Joint Angles
```python
import json

with open('your_hand_image_pose.json', 'r') as f:
    data = json.load(f)

# Get Euler angles for specific joints (in degrees)
thumb_angles = data['joint_rotations']['THUMB_MCP']['euler_xyz_deg']
index_angles = data['joint_rotations']['INDEX_PIP']['euler_xyz_deg']

print(f"Thumb MCP: Roll={thumb_angles[0]:.1f}°, Pitch={thumb_angles[1]:.1f}°, Yaw={thumb_angles[2]:.1f}°")
```

### Example: Send to Robot
```python
# Pseudo-code for robot control
for joint_name, rotation_data in data['joint_rotations'].items():
    angles = rotation_data['euler_xyz_deg']
    robot.set_joint_angles(joint_name, angles)
```

---

## 5. Tips for Best Results

### Image Requirements
- ✓ Hand should be clearly visible
- ✓ Good lighting (avoid harsh shadows)
- ✓ Hand occupies >15% of image
- ✓ Any orientation (palm, back, side view)

### If Detection Fails
1. Try with better lighting
2. Ensure hand is not too small in frame
3. Reduce background clutter
4. Check that image file is valid

### Improve Accuracy
In `infer_hand_pose.py`, line 44, change:
```python
model_complexity=2  # Instead of 1 (slower but more accurate)
```

### Improve Speed
```python
model_complexity=0  # Instead of 1 (faster but less accurate)
```

---

## 6. Gesture Reference

Based on your provided gesture table, you can test with these poses:
- Lateral prehension (LP)
- Thumb adduction (TA)
- Index finger extension (IFE)
- Hand open (HO)
- Hand close (HC)
- And all others from your table

The model will estimate joint angles even for occluded fingers!

---

## 7. Common Issues

### "No hand detected"
→ Hand may be too small or unclear in image  
→ Try cropping closer to the hand

### Import errors
→ Activate virtual environment: `.\venv\Scripts\Activate.ps1`  
→ Reinstall: `pip install -r requirements.txt`

### Slow inference
→ First run downloads MediaPipe models (~20MB)  
→ Subsequent runs are faster

---

## 8. Next Steps

Once this works:
1. Process multiple gesture images from your table
2. Build a dataset of gesture → joint angles
3. Use for robot training or direct control
4. Implement temporal smoothing for video sequences

**Questions?** Check README.md for detailed documentation.
