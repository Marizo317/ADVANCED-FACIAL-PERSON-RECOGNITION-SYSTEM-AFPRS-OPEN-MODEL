# vision/app/core/face_analyzer.py
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Tuple, List
from config import CFG

@dataclass
class FaceAnalysis:
    landmarks_468: Optional[np.ndarray] = None
    iris_left: Optional[np.ndarray] = None
    iris_right: Optional[np.ndarray] = None
    head_pose: Optional[Tuple[float, float, float]] = None
    gaze: Optional[str] = None

class FaceAnalyzer:
    def __init__(self):
        self.mp_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=CFG.MESH_REFINE,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Key indices
        self.NOSE = 1
        self.CHIN = 152
        self.L_EYE = 33
        self.R_EYE = 263
        self.L_MOUTH = 61
        self.R_MOUTH = 291
        self.L_IRIS = list(range(468, 473))
        self.R_IRIS = list(range(473, 478))
        print("[FACE_ANALYZER] MediaPipe FaceMesh ready")
    
    def analyze(self, frame: np.ndarray, bbox: List[int]) -> Optional[FaceAnalysis]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Expand ROI
        pad = int((x2 - x1) * 0.3)
        x1p, y1p = max(0, x1-pad), max(0, y1-pad)
        x2p, y2p = min(w, x2+pad), min(h, y2+pad)
        
        roi = frame[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            return None
        
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        lms = results.multi_face_landmarks[0]
        rh, rw = roi.shape[:2]
        
        # 468 landmarks
        lms_468 = np.array([[l.x * rw + x1p, l.y * rh + y1p] for l in lms.landmark[:468]])
        
        # Iris
        iris_l = iris_r = None
        if len(lms.landmark) > 468:
            iris_l = np.array([[lms.landmark[i].x * rw + x1p, lms.landmark[i].y * rh + y1p] for i in self.L_IRIS])
            iris_r = np.array([[lms.landmark[i].x * rw + x1p, lms.landmark[i].y * rh + y1p] for i in self.R_IRIS])
        
        # Head pose
        pose = self._head_pose(lms_468, (h, w))
        gaze = self._gaze(pose)
        
        return FaceAnalysis(lms_468, iris_l, iris_r, pose, gaze)
    
    def _head_pose(self, lms, shape) -> Tuple[float, float, float]:
        h, w = shape
        img_pts = np.array([lms[self.NOSE], lms[self.CHIN], lms[self.L_EYE], 
                           lms[self.R_EYE], lms[self.L_MOUTH], lms[self.R_MOUTH]], dtype=np.float64)
        
        model_pts = np.array([
            (0, 0, 0), (0, -330, -65), (-225, 170, -135),
            (225, 170, -135), (-150, -150, -125), (150, -150, -125)
        ], dtype=np.float64)
        
        cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        
        ok, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam, np.zeros((4,1)))
        if not ok:
            return (0, 0, 0)
        
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        
        if sy > 1e-6:
            pitch = np.arctan2(rmat[2,1], rmat[2,2])
            yaw = np.arctan2(-rmat[2,0], sy)
            roll = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            pitch = np.arctan2(-rmat[1,2], rmat[1,1])
            yaw = np.arctan2(-rmat[2,0], sy)
            roll = 0
        
        return (float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll)))
    
    def _gaze(self, pose) -> str:
        if not pose:
            return "unknown"
        yaw, pitch, _ = pose
        if abs(yaw) < 15 and abs(pitch) < 15:
            return "forward"
        if yaw > 20:
            return "left"
        if yaw < -20:
            return "right"
        if pitch > 15:
            return "down"
        if pitch < -15:
            return "up"
        return "forward"
