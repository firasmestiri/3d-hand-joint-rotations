# Release Notes

## Version 2.0 - Windows Compatibility Fix (December 16, 2025)

### 🔧 Critical Fix: MediaPipe Windows Compatibility

**Problem:** MediaPipe 0.10.30+ has a Windows-specific bug causing `function 'free' not found` error.

**Solution:** Locked to MediaPipe 0.10.9, which is stable and fully functional on Windows.

---

### ✅ What Was Fixed

1. **MediaPipe Version Lock**
   - Updated `requirements.txt` to specify `mediapipe==0.10.9`
   - Prevents automatic upgrade to buggy versions

2. **New Working Script**
   - Created `infer_hand_pose_v2.py` with simplified, Windows-compatible code
   - Removed complex API version detection
   - Direct implementation using MediaPipe 0.10.9 Tasks API

3. **Updated All Documentation**
   - `README.md` - Added version warning, updated all commands
   - `QUICKSTART.md` - Updated script name to v2
   - `SUMMARY.md` - Added version compatibility notes
   - `test_installation.py` - Now checks MediaPipe version
   - `setup.ps1` - Updated to reference v2 script

4. **New Documentation Files**
   - `KNOWN_ISSUES.md` - Documents the MediaPipe issue and solution
   - `QUICK_REFERENCE.md` - One-page quick reference
   - `RELEASE_NOTES.md` - This file

---

### 📋 Changes Summary

#### Modified Files
- ✏️ `requirements.txt` - Pinned MediaPipe to 0.10.9
- ✏️ `README.md` - Added version warning, updated usage examples
- ✏️ `QUICKSTART.md` - Updated all script references to v2
- ✏️ `SUMMARY.md` - Updated file manifest and version info
- ✏️ `test_installation.py` - Added MediaPipe version check
- ✏️ `setup.ps1` - Updated final instructions

#### New Files
- ✨ `infer_hand_pose_v2.py` - Windows-compatible main script
- ✨ `KNOWN_ISSUES.md` - Issue documentation
- ✨ `QUICK_REFERENCE.md` - Quick reference card
- ✨ `RELEASE_NOTES.md` - This file

#### Unchanged Files
- 📄 `infer_hand_pose.py` - Kept for reference (legacy)
- 📄 `.gitignore` - No changes needed

---

### 🎯 Impact for New Users

**Before this fix:**
- Users would encounter cryptic errors on Windows
- Manual intervention required to downgrade MediaPipe
- Documentation referenced non-working version

**After this fix:**
- Automatic installation of correct version
- No manual intervention needed
- Clear documentation pointing to working script
- Comprehensive troubleshooting guide

---

### 🧪 Testing Performed

All functionality verified on Windows 11, Python 3.11.9:

✅ Installation via `pip install -r requirements.txt`  
✅ Setup script (`setup.ps1`)  
✅ Installation tests (`test_installation.py`)  
✅ Hand detection on test image (TLFE_gest.png)  
✅ Joint rotation computation  
✅ JSON output generation  
✅ Visualization generation  
✅ All 21 joints outputting correct data  

**Test Results:** 4/4 tests passed, full inference successful

---

### 📦 Recommended Setup (Post-Fix)

```powershell
# Clone or download project
cd "3d hand pose estimation"

# Automated setup (recommended)
.\setup.ps1

# Manual setup (alternative)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_installation.py

# Run inference
python infer_hand_pose_v2.py your_image.jpg --visualize
```

---

### 🔮 Future Considerations

**When MediaPipe fixes Windows compatibility:**
1. Test newer versions as they're released
2. Update `requirements.txt` if a working version is found
3. Consider merging v2 improvements into v1
4. Update documentation accordingly

**Current Recommendation:**
Stay with MediaPipe 0.10.9 until Windows compatibility is confirmed in newer releases.

---

### 🐛 Known Limitations

1. **MediaPipe Version Locked**
   - Cannot use features from MediaPipe 0.10.10+
   - This is acceptable as 0.10.9 has all needed functionality

2. **Two Script Versions**
   - Users might be confused about which to use
   - Mitigated by clear documentation and warnings

3. **Platform-Specific Issue**
   - Linux/macOS users might not need the version lock
   - But using 0.10.9 works fine on all platforms

---

### 📝 Migration Guide

**If you have an existing installation with MediaPipe 0.10.30+:**

```powershell
# Activate your environment
.\venv\Scripts\Activate.ps1

# Remove problematic version
pip uninstall mediapipe

# Install working version
pip install mediapipe==0.10.9

# Verify
python -c "import mediapipe as mp; print(mp.__version__)"
# Should print: 0.10.9

# Test
python test_installation.py

# Use v2 script
python infer_hand_pose_v2.py your_image.jpg --visualize
```

---

### ✨ Benefits

1. **Plug-and-Play** - Works immediately after setup
2. **No Debug Time** - Users won't waste hours troubleshooting
3. **Production Ready** - Stable, tested version
4. **Clear Documentation** - Multiple guides for different needs
5. **Future-Proof** - Clear upgrade path when MediaPipe is fixed

---

### 🙏 Acknowledgments

Issue discovered and fixed during initial testing on Windows 11.  
Thanks to the MediaPipe team for the excellent hand tracking model.

---

**Status:** ✅ Production Ready  
**Tested On:** Windows 11, Python 3.11.9  
**MediaPipe Version:** 0.10.9  
**Last Updated:** December 16, 2025
