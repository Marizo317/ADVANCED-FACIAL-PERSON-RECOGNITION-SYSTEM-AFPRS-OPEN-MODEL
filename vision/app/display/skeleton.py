# vision/app/display/skeleton.py
import cv2
from config import CFG
from core.person_detector import SKELETON

class SkeletonViz:
    def draw(self, frame, keypoints, thresh=0.3):
        if keypoints is None:
            return frame
        
        # Lines
        for (i, j) in SKELETON:
            if keypoints[i][2] > thresh and keypoints[j][2] > thresh:
                p1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                p2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, p1, p2, CFG.C_SKEL, 2, cv2.LINE_AA)
        
        # Points
        for kp in keypoints:
            if kp[2] > thresh:
                pt = (int(kp[0]), int(kp[1]))
                cv2.circle(frame, pt, 4, CFG.C_SKEL, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 2, (255,255,255), -1, cv2.LINE_AA)
        
        return frame
