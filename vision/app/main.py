# vision/app/main.py
import cv2
import time
from config import CFG
from core.capture import Capture
from core.person_detector import PersonDetector
from core.face_detector import FaceDetector
from core.associator import Associator
from core.tracker import Tracker
from core.face_analyzer import FaceAnalyzer
from core.person_analyzer import PersonAnalyzer
from core.llm_client import LLMClient
from display.overlays import Overlays
from display.skeleton import SkeletonViz
from display.face_mesh import FaceMeshViz
from display.hud import HUD
from display.panel import Panel

class AFPRS:
    def __init__(self):
        print("="*60)
        print("  AFPRS - Advanced Facial & Person Recognition System")
        print("="*60)
        
        # Core
        self.cap = Capture()
        self.person_det = PersonDetector()
        self.face_det = FaceDetector()
        self.assoc = Associator()
        self.tracker = Tracker()
        self.face_anal = FaceAnalyzer()
        self.person_anal = PersonAnalyzer()
        self.llm = LLMClient()
        
        # Display
        self.overlays = Overlays()
        self.skel = SkeletonViz()
        self.mesh = FaceMeshViz()
        self.hud = HUD()
        self.panel = Panel()
        
        # State
        self.frame_count = 0
        self.show = {'person': True, 'face': True, 'skel': True, 'mesh': True, 'traj': True}
        
        cv2.namedWindow(CFG.WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("\nControls: q=quit, p/f/s/m/t=toggle overlays")
        print("="*60 + "\n")
    
    def run(self):
        while True:
            ret, pkt = self.cap.read()
            if not ret:
                continue
            
            frame = pkt.frame
            self.frame_count += 1
            timings = {}
            
            # Detection
            t0 = time.perf_counter()
            persons = self.person_det.detect(frame)
            timings['person'] = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            faces = self.face_det.detect(frame)
            timings['face'] = time.perf_counter() - t0
            
            # Association & Tracking
            assocs = self.assoc.associate(persons, faces)
            tracks = self.tracker.update(assocs)
            
            # Analysis
            if self.frame_count % CFG.ANALYSIS_INTERVAL == 0:
                for t in tracks:
                    if t.face_bbox:
                        fa = self.face_anal.analyze(frame, t.face_bbox)
                        if fa:
                            t.landmarks_468 = fa.landmarks_468
                            t.iris_left = fa.iris_left
                            t.iris_right = fa.iris_right
                            t.head_pose = fa.head_pose
                    
                    if t.person_bbox:
                        pa = self.person_anal.analyze(frame, t.person_bbox)
                        t.clothing_upper = pa.upper_color
                        t.clothing_lower = pa.lower_color
                        t.position_zone = pa.position_zone
                    
                    self.llm.request(t)
                    desc = self.llm.get(t.id)
                    if desc:
                        t.description = desc
            
            # Render
            disp = frame.copy()
            for t in tracks:
                if self.show['traj']:
                    disp = self.overlays.draw_trajectory(disp, t)
                if self.show['person']:
                    disp = self.overlays.draw_person(disp, t)
                if self.show['skel'] and t.keypoints is not None:
                    disp = self.skel.draw(disp, t.keypoints)
                if self.show['face']:
                    disp = self.overlays.draw_face(disp, t)
                if self.show['mesh'] and t.landmarks_468 is not None:
                    disp = self.mesh.draw_contours(disp, t.landmarks_468)
                    disp = self.mesh.draw_iris(disp, t.iris_left, t.iris_right)
            
            self.hud.update(timings)
            n_p = sum(1 for t in tracks if t.person_bbox)
            n_f = sum(1 for t in tracks if t.face_bbox)
            disp = self.hud.draw(disp, n_p, n_f)
            disp = self.panel.draw(disp, tracks)
            
            cv2.imshow(CFG.WINDOW_NAME, disp)
            
            # Input
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                self.show['person'] = not self.show['person']
            elif k == ord('f'):
                self.show['face'] = not self.show['face']
            elif k == ord('s'):
                self.show['skel'] = not self.show['skel']
            elif k == ord('m'):
                self.show['mesh'] = not self.show['mesh']
            elif k == ord('t'):
                self.show['traj'] = not self.show['traj']
        
        self.cap.release()
        self.llm.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    AFPRS().run()
