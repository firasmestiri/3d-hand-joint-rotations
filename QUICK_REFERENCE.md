# Quick Reference Card

## ⚡ TL;DR

```powershell
# Setup (once)
.\setup.ps1

# Run inference
python infer_hand_pose_v2.py your_image.jpg --visualize

# Output: your_image_pose.json + your_image_visualization.jpg
```

---

## 📋 Important Notes

### ✅ Use This Script
`infer_hand_pose_v2.py` - Windows-compatible, tested and working

### 📦 Correct Version
MediaPipe 0.10.9 (specified in requirements.txt)

### 🚫 Common Mistakes
- ❌ Don't use `infer_hand_pose.py` (old version)
- ❌ Don't install `mediapipe>=0.10.30` (has Windows bugs)
- ❌ Don't forget to activate venv: `.\venv\Scripts\Activate.ps1`

---

## 🔧 Troubleshooting One-Liners

```powershell
# Check MediaPipe version (should be 0.10.9)
python -c "import mediapipe as mp; print(mp.__version__)"

# Fix wrong version
pip uninstall mediapipe; pip install mediapipe==0.10.9

# Test everything is working
python test_installation.py

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

---

## 📁 File Structure

```
📂 Project Root
├── 📄 infer_hand_pose_v2.py     ← USE THIS (main script)
├── 📄 requirements.txt           ← Dependencies (MediaPipe 0.10.9)
├── 📄 setup.ps1                  ← Automated setup
├── 📄 test_installation.py       ← Verify setup
├── 📄 README.md                  ← Full documentation
├── 📄 QUICKSTART.md              ← Quick start guide
├── 📄 KNOWN_ISSUES.md            ← Version compatibility info
└── 📄 QUICK_REFERENCE.md         ← This file
```

---

## 🎯 Output Format

Each JSON file contains for **21 joints**:
- `euler_xyz_deg` - Roll, Pitch, Yaw (degrees) → Use for servos
- `quaternion_wxyz` - [w,x,y,z] → Use for smooth control
- `axis_angle` - Axis + angle → Use for revolute joints
- `position_3d_m` - 3D position (meters)

---

## 🔗 Links

- Setup: Run `.\setup.ps1`
- Full Docs: [README.md](README.md)
- Quick Start: [QUICKSTART.md](QUICKSTART.md)
- Issues: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
- Project Summary: [SUMMARY.md](SUMMARY.md)

---

## 💡 Pro Tips

1. **First-time users**: Just run `.\setup.ps1` and follow prompts
2. **Hand not detected?** Ensure hand occupies >15% of image frame
3. **Need higher accuracy?** Adjust `min_detection_confidence` in script
4. **Processing multiple images?** Write a loop or batch script
5. **For research**: Use JSON output directly in your analysis pipeline

---

**Last Updated:** December 16, 2025  
**Tested On:** Windows 11, Python 3.11.9, MediaPipe 0.10.9  
**Status:** ✅ Fully Working
