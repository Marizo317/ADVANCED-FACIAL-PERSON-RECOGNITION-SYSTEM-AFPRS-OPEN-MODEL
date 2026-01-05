# vision/app/core/face_detector.py
import torch
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional
import time
from config import CFG

@dataclass
class FaceDet:
    bbox: List[int]
    confidence: float
    landmarks_5: Optional[np.ndarray] = None

class FaceDetector:
    def __init__(self):
        self.model = YOLO(CFG.FACE_MODEL)
        self.model.to(CFG.DEVICE)
        if CFG.USE_FP16:
            self.model.model.half()
        self._warmup()
        self.inf_time = 0
        print(f"[FACE_DET] Loaded: {CFG.FACE_MODEL}")
    
    def _warmup(self):
        dummy = np.zeros((CFG.FRAME_HEIGHT, CFG.FRAME_WIDTH, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(dummy, imgsz=CFG.FACE_IMG_SIZE, verbose=False)
    
    def detect(self, frame: np.ndarray) -> List[FaceDet]:
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=CFG.FACE_IMG_SIZE,
            conf=CFG.FACE_CONF,
            verbose=False
        )[0]
        self.inf_time = time.perf_counter() - t0
        
        dets = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            lmks = results.keypoints.xy.cpu().numpy() if hasattr(results, 'keypoints') and results.keypoints else None
            
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                dets.append(FaceDet(
                    bbox=box.astype(int).tolist(),
                    confidence=float(conf),
                    landmarks_5=lmks[i] if lmks is not None else None
                ))
        return dets
