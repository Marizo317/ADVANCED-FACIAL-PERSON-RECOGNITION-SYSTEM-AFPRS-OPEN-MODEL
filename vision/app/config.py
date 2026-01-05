# vision/app/config.py
import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FP16: bool = True
    
    # Capture
    CAMERA_ID: int = 0
    FRAME_WIDTH: int = 1920
    FRAME_HEIGHT: int = 1080
    TARGET_FPS: int = 60
    
    # Person Detection
    PERSON_MODEL: str = "models/yolo11m-pose.pt"
    PERSON_CONF: float = 0.5
    PERSON_IOU: float = 0.45
    PERSON_IMG_SIZE: int = 640
    
    # Face Detection
    FACE_MODEL: str = "models/yolo11n-face.pt"
    FACE_CONF: float = 0.5
    FACE_IMG_SIZE: int = 640
    
    # Tracking
    TRACK_BUFFER: int = 30
    MATCH_THRESH: float = 0.8
    TRAJECTORY_LEN: int = 60
    
    # Face Analysis
    MESH_REFINE: bool = True  # 478 puntos con iris
    ANALYSIS_INTERVAL: int = 3
    
    # LLM Service
    LLM_URL: str = os.getenv("LLM_SERVICE_URL", "http://afprs-llm:8001")
    LLM_TIMEOUT: float = 5.0
    DESC_INTERVAL: float = 2.5
    
    # Display
    WINDOW_NAME: str = "AFPRS - Advanced Facial & Person Recognition"
    PANEL_HEIGHT: int = 180
    HUD_WIDTH: int = 280
    
    # Colors (BGR)
    C_HIGH: tuple = (46, 204, 113)
    C_MED: tuple = (241, 196, 15)
    C_LOW: tuple = (231, 76, 60)
    C_PERSON: tuple = (255, 165, 0)
    C_FACE: tuple = (0, 255, 255)
    C_SKEL: tuple = (46, 204, 113)
    C_TRAJ: tuple = (52, 152, 219)
    C_MESH: tuple = (155, 89, 182)
    C_IRIS: tuple = (26, 188, 156)
    C_TEXT: tuple = (220, 220, 230)
    C_BG: tuple = (15, 15, 20)

CFG = Config()
