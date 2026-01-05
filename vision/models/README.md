# Models Directory

This directory should contain the YOLO models:

- `yolo11m-pose.pt` - Person detection with pose estimation
- `yolo11n-face.pt` - Face detection model

## Download Models

Run the download script:

**Linux/WSL:**
```bash
./scripts/download_models.sh
```

**Windows PowerShell:**
```powershell
.\scripts\download_models.ps1
```

## Manual Download

If the scripts don't work, download manually:

1. **YOLO11m-pose**: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m-pose.pt
2. **YOLO11n (base)**: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n.pt

Rename `yolo11n.pt` to `yolo11n-face.pt` for the face detector.

Note: For better face detection, consider training a custom YOLO11 model on a face dataset like WIDER Face.
