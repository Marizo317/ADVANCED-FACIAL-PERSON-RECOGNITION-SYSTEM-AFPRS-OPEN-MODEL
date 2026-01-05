# vision/app/display/overlays.py
import cv2
import numpy as np
from config import CFG

class Overlays:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_person(self, frame, track):
        if not track.person_bbox:
            return frame
        x1, y1, x2, y2 = track.person_bbox
        
        c = self._conf_color(track.person_conf)
        th = max(1, int(track.person_conf * 3))
        
        cv2.rectangle(frame, (x1,y1), (x2,y2), c, th)
        
        # Corners
        cl = 15
        cv2.line(frame, (x1,y1), (x1+cl,y1), c, th+1)
        cv2.line(frame, (x1,y1), (x1,y1+cl), c, th+1)
        cv2.line(frame, (x2,y1), (x2-cl,y1), c, th+1)
        cv2.line(frame, (x2,y1), (x2,y1+cl), c, th+1)
        cv2.line(frame, (x1,y2), (x1+cl,y2), c, th+1)
        cv2.line(frame, (x1,y2), (x1,y2-cl), c, th+1)
        cv2.line(frame, (x2,y2), (x2-cl,y2), c, th+1)
        cv2.line(frame, (x2,y2), (x2,y2-cl), c, th+1)
        
        # Label
        lbl = f"ID:{track.id} [{track.person_conf:.0%}]"
        self._label(frame, lbl, (x1, y1-5), c)
        return frame
    
    def draw_face(self, frame, track):
        if not track.face_bbox:
            return frame
        x1, y1, x2, y2 = track.face_bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), CFG.C_FACE, 1)
        
        if track.age and track.gender:
            self._label(frame, f"{track.gender}, ~{track.age}y", (x1, y2+12), CFG.C_FACE, 0.35)
        if track.emotion:
            self._label(frame, track.emotion, (x1, y2+24), CFG.C_FACE, 0.35)
        return frame
    
    def draw_trajectory(self, frame, track):
        pts = list(track.trajectory)
        if len(pts) < 2:
            return frame
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            th = max(1, int(alpha * 2))
            cv2.line(frame, pts[i-1], pts[i], CFG.C_TRAJ, th, cv2.LINE_AA)
        return frame
    
    def _conf_color(self, conf):
        if conf > 0.75:
            return CFG.C_HIGH
        if conf > 0.5:
            return CFG.C_MED
        return CFG.C_LOW
    
    def _label(self, frame, text, pos, color, scale=0.4):
        (w, h), _ = cv2.getTextSize(text, self.font, scale, 1)
        x, y = pos
        cv2.rectangle(frame, (x, y-h-4), (x+w+4, y+2), CFG.C_BG, -1)
        cv2.putText(frame, text, (x+2, y-2), self.font, scale, color, 1, cv2.LINE_AA)
