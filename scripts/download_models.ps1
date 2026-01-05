# scripts/download_models.ps1 - PowerShell version for Windows
Write-Host "Creating models directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "vision/models" | Out-Null

Write-Host "Downloading YOLO11 models..." -ForegroundColor Green

# Person + Pose model
$poseUrl = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m-pose.pt"
$poseDest = "vision/models/yolo11m-pose.pt"
Write-Host "Downloading pose model..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $poseUrl -OutFile $poseDest

# Face model (using base model as placeholder)
$faceUrl = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n.pt"
$faceDest = "vision/models/yolo11n-face.pt"
Write-Host "Downloading face model..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $faceUrl -OutFile $faceDest

Write-Host "Models downloaded!" -ForegroundColor Green
