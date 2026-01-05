# vision/app/display/hud.py
import cv2
import numpy as np
from collections import deque
import time
from config import CFG

class HUD:
    def __init__(self):
        self.fps_hist = deque(maxlen=30)
        self.last_t = time.time()
        self.fps = 0
        self.timings = {}
    
    def update(self, timings=None):
        now = time.time()
        dt = now - self.last_t
        if dt > 0:
            self.fps_hist.append(1/dt)
            self.fps = np.mean(self.fps_hist)
        self.last_t = now
        if timings:
            self.timings = timings
    
    def draw(self, frame, n_persons=0, n_faces=0):
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (10+CFG.HUD_WIDTH, 10+160), CFG.C_BG, -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        cv2.rectangle(frame, (10,10), (10+CFG.HUD_WIDTH, 10+160), CFG.C_HIGH, 1)
        
        # Title
        cv2.putText(frame, "AFPRS STATUS", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CFG.C_HIGH, 1)
        cv2.line(frame, (20,42), (CFG.HUD_WIDTH,42), CFG.C_HIGH, 1)
        
        # Metrics
        y = 65
        fps_c = CFG.C_HIGH if self.fps >= 30 else (CFG.C_MED if self.fps >= 15 else CFG.C_LOW)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_c, 1)
        y += 22
        cv2.putText(frame, f"Persons: {n_persons}", (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CFG.C_TEXT, 1)
        y += 22
        cv2.putText(frame, f"Faces: {n_faces}", (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CFG.C_TEXT, 1)
        y += 22
        
        if self.timings:
            total = sum(self.timings.values()) * 1000
            lat_c = CFG.C_HIGH if total < 33 else (CFG.C_MED if total < 66 else CFG.C_LOW)
            cv2.putText(frame, f"Latency: {total:.1f}ms", (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lat_c, 1)
        
        # Status
        cv2.putText(frame, "● OPERATIONAL", (20,155), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CFG.C_HIGH, 1)
        return frame
