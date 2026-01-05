# vision/app/core/person_analyzer.py
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import Counter

@dataclass  
class PersonAnalysis:
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    position_zone: Optional[str] = None
    depth: Optional[str] = None

class PersonAnalyzer:
    COLORS = {
        'red': ([0,100,100], [10,255,255]),
        'orange': ([10,100,100], [25,255,255]),
        'yellow': ([25,100,100], [35,255,255]),
        'green': ([35,100,100], [85,255,255]),
        'blue': ([85,100,100], [125,255,255]),
        'purple': ([125,100,100], [155,255,255]),
        'white': ([0,0,200], [180,30,255]),
        'black': ([0,0,0], [180,255,50]),
        'gray': ([0,0,50], [180,30,200]),
    }
    
    def analyze(self, frame: np.ndarray, bbox: List[int]) -> PersonAnalysis:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return PersonAnalysis()
        
        h, w = roi.shape[:2]
        fh, fw = frame.shape[:2]
        
        # Upper clothing
        upper = roi[int(h*0.15):int(h*0.45), int(w*0.2):int(w*0.8)]
        upper_c = self._classify_color(upper) if upper.size > 0 else None
        
        # Lower clothing  
        lower = roi[int(h*0.5):int(h*0.85), int(w*0.2):int(w*0.8)]
        lower_c = self._classify_color(lower) if lower.size > 0 else None
        
        # Position
        cx = (x1 + x2) / 2
        zone = "left" if cx < fw * 0.33 else ("right" if cx > fw * 0.66 else "center")
        
        # Depth
        area_ratio = ((x2-x1) * (y2-y1)) / (fw * fh)
        depth = "near" if area_ratio > 0.15 else ("far" if area_ratio < 0.05 else "mid")
        
        return PersonAnalysis(upper_c, lower_c, zone, depth)
    
    def _classify_color(self, roi) -> str:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        best, best_count = "unknown", 0
        for name, (lo, hi) in self.COLORS.items():
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            count = cv2.countNonZero(mask)
            if count > best_count:
                best_count = count
                best = name
        return best
