#!/bin/bash
# scripts/download_models.sh
mkdir -p vision/models

echo "Downloading YOLO11 models..."
# Person + Pose
wget -O vision/models/yolo11m-pose.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m-pose.pt

# Face (usar modelo face específico o entrenar)
wget -O vision/models/yolo11n-face.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n.pt

echo "Models downloaded!"
