# vision/app/display/panel.py
import cv2
from datetime import datetime
from config import CFG

class Panel:
    def draw(self, frame, tracks):
        h, w = frame.shape[:2]
        py = h - CFG.PANEL_HEIGHT
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,py), (w,h), CFG.C_BG, -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        cv2.line(frame, (0,py), (w,py), CFG.C_HIGH, 2)
        
        # Title
        cv2.putText(frame, "AI-GENERATED DESCRIPTIONS", (20,py+25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CFG.C_TEXT, 1)
        
        # Descriptions
        y = py + 55
        described = [t for t in tracks if t.description][:3]
        
        if not described:
            cv2.putText(frame, "Analyzing detected persons...", (30,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,130), 1)
        else:
            for t in described:
                ts = datetime.fromtimestamp(t.last_seen).strftime("%H:%M:%S")
                header = f"[{ts}] Person #{t.id}"
                c = CFG.C_HIGH if t.get_conf() > 0.75 else (CFG.C_MED if t.get_conf() > 0.5 else CFG.C_LOW)
                
                cv2.putText(frame, header, (30,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                y += 20
                
                # Wrap description
                desc = t.description[:90]
                cv2.putText(frame, desc, (50,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CFG.C_TEXT, 1)
                y += 30
        
        return frame
