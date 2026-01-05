# vision/app/display/face_mesh.py
import cv2
import numpy as np
import mediapipe as mp
from config import CFG

class FaceMeshViz:
    def __init__(self):
        self.mp_mesh = mp.solutions.face_mesh
        self.OVAL = list(self.mp_mesh.FACEMESH_FACE_OVAL)
        self.L_EYE = list(self.mp_mesh.FACEMESH_LEFT_EYE)
        self.R_EYE = list(self.mp_mesh.FACEMESH_RIGHT_EYE)
        self.LIPS = list(self.mp_mesh.FACEMESH_LIPS)
    
    def draw_contours(self, frame, lms):
        if lms is None:
            return frame
        self._draw(frame, lms, self.OVAL, CFG.C_MESH)
        self._draw(frame, lms, self.L_EYE, (52,152,219))
        self._draw(frame, lms, self.R_EYE, (52,152,219))
        self._draw(frame, lms, self.LIPS, (231,76,60))
        return frame
    
    def draw_iris(self, frame, iris_l, iris_r):
        for iris in [iris_l, iris_r]:
            if iris is not None and len(iris) > 0:
                c = (int(iris[0][0]), int(iris[0][1]))
                r = int(np.linalg.norm(iris[0] - iris[1])) if len(iris) > 1 else 5
                cv2.circle(frame, c, r, CFG.C_IRIS, 1, cv2.LINE_AA)
                cv2.circle(frame, c, 2, CFG.C_IRIS, -1)
        return frame
    
    def draw_pose_axes(self, frame, lms, pose, length=50):
        if lms is None or pose is None:
            return frame
        origin = (int(lms[1][0]), int(lms[1][1]))
        yaw, pitch, roll = pose
        
        # Simplified axes
        cv2.arrowedLine(frame, origin, (origin[0]+length, origin[1]), (0,0,255), 2, tipLength=0.3)
        cv2.arrowedLine(frame, origin, (origin[0], origin[1]-length), (0,255,0), 2, tipLength=0.3)
        cv2.arrowedLine(frame, origin, (origin[0]+int(length*np.sin(np.radians(yaw))), origin[1]), (255,0,0), 2, tipLength=0.3)
        return frame
    
    def _draw(self, frame, lms, conns, color):
        for (i, j) in conns:
            if i < len(lms) and j < len(lms):
                p1 = (int(lms[i][0]), int(lms[i][1]))
                p2 = (int(lms[j][0]), int(lms[j][1]))
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
