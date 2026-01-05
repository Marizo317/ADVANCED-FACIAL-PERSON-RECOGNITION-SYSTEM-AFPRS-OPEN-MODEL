# vision/app/core/person_detector.py
import torch
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional
import time
from config import CFG

@dataclass
class PersonDet:
    bbox: List[int]
    confidence: float
    keypoints: Optional[np.ndarray] = None

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),           # Head
    (5,6),(5,7),(7,9),(6,8),(8,10),    # Arms
    (5,11),(6,12),(11,12),             # Torso
    (11,13),(13,15),(12,14),(14,16)    # Legs
]

class PersonDetector:
    def __init__(self):
        self.model = YOLO(CFG.PERSON_MODEL)
        self.model.to(CFG.DEVICE)
        if CFG.USE_FP16:
            self.model.model.half()
        self._warmup()
        self.inf_time = 0
        print(f"[PERSON_DET] Loaded: {CFG.PERSON_MODEL}")
    
    def _warmup(self):
        dummy = np.zeros((CFG.FRAME_HEIGHT, CFG.FRAME_WIDTH, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(dummy, imgsz=CFG.PERSON_IMG_SIZE, verbose=False)
    
    def detect(self, frame: np.ndarray) -> List[PersonDet]:
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=CFG.PERSON_IMG_SIZE,
            conf=CFG.PERSON_CONF,
            iou=CFG.PERSON_IOU,
            classes=[0],
            verbose=False
        )[0]
        self.inf_time = time.perf_counter() - t0
        
        dets = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            kpts = results.keypoints.data.cpu().numpy() if hasattr(results, 'keypoints') and results.keypoints else None
            
            for i, (box, conf) in enumerate(zip(boxes, confs)):
                dets.append(PersonDet(
                    bbox=box.astype(int).tolist(),
                    confidence=float(conf),
                    keypoints=kpts[i] if kpts is not None else None
                ))
        return dets
