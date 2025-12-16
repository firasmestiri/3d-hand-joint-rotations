# Known Issues and Solutions

## MediaPipe Version Compatibility (RESOLVED)

### Issue
MediaPipe versions 0.10.30+ have a critical bug on Windows systems:
- Error: `function 'free' not found`
- Error: `module 'mediapipe' has no attribute 'solutions'`
- Caused by C library binding issues in the new Tasks API

### Solution
✅ **Fixed**: Project now uses MediaPipe 0.10.9

### What We Did
1. **requirements.txt** - Pinned to `mediapipe==0.10.9`
2. **infer_hand_pose_v2.py** - Created Windows-compatible version
3. **All documentation** - Updated to reference v2 script

### For New Users
Just follow the normal setup process:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python infer_hand_pose_v2.py your_image.jpg --visualize
```

The correct MediaPipe version will be installed automatically.

---

## Script Versions

### Use This: `infer_hand_pose_v2.py` ✅
- **Status**: Fully working
- **MediaPipe**: 0.10.9 compatible
- **Platform**: Windows, macOS, Linux
- **Features**: All functionality working

### Legacy: `infer_hand_pose.py` 
- **Status**: May have issues on Windows with newer MediaPipe
- **Kept for**: Reference and compatibility with older setups
- **Recommendation**: Use v2 instead

---

## Version History

### v2.0 (Current)
- Fixed MediaPipe compatibility for Windows
- Pinned MediaPipe to 0.10.9
- Simplified initialization code
- Enhanced error messages

### v1.0 (Original)
- Initial implementation
- Attempted to support both old and new MediaPipe APIs
- Had compatibility issues with MediaPipe 0.10.30+

---

## Testing

All functionality has been tested and verified:
- ✅ Hand detection working
- ✅ 3D landmark estimation working
- ✅ Joint rotation computation working
- ✅ Visualization working
- ✅ JSON output working
- ✅ Windows compatibility confirmed

Test image: `TLFE_gest.png` (Thumb and Little Finger Extension gesture)

---

## Future Considerations

If MediaPipe releases a Windows-compatible version in the future:
1. Test with newer version: `pip install mediapipe==<new_version>`
2. If it works, update `requirements.txt`
3. Consider deprecating v2 in favor of updated v1

Until then, MediaPipe 0.10.9 is the stable, recommended version.

---

## Getting Help

If you encounter issues:
1. **Check MediaPipe version**: `python -c "import mediapipe as mp; print(mp.__version__)"`
2. **Should show**: `0.10.9`
3. **If different**: `pip uninstall mediapipe && pip install mediapipe==0.10.9`
4. **Verify setup**: `python test_installation.py`
5. **Use correct script**: `infer_hand_pose_v2.py` (not v1)

---

Last updated: December 16, 2025
