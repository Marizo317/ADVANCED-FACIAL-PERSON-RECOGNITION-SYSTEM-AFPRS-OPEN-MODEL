# vision/app/core/capture.py
import cv2
import time
from threading import Thread, Event
from queue import Queue
from dataclasses import dataclass
from config import CFG

@dataclass
class FramePacket:
    frame: any
    timestamp: float
    frame_id: int

class Capture:
    def __init__(self, source=None):
        src = source if source is not None else CFG.CAMERA_ID
        self.cap = cv2.VideoCapture(src)
        self._configure()
        
        self.queue = Queue(maxsize=2)
        self.running = Event()
        self.running.set()
        self.frame_count = 0
        self.fps = 0
        self._fps_time = time.time()
        self._fps_count = 0
        
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[CAPTURE] Started: {CFG.FRAME_WIDTH}x{CFG.FRAME_HEIGHT}")
    
    def _configure(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CFG.TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def _loop(self):
        while self.running.is_set():
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                self._fps_count += 1
                
                if self.queue.full():
                    try: self.queue.get_nowait()
                    except: pass
                
                self.queue.put(FramePacket(frame, time.time(), self.frame_count))
                
                # FPS calc
                elapsed = time.time() - self._fps_time
                if elapsed >= 1.0:
                    self.fps = self._fps_count / elapsed
                    self._fps_count = 0
                    self._fps_time = time.time()
    
    def read(self):
        if not self.queue.empty():
            return True, self.queue.get()
        return False, None
    
    def release(self):
        self.running.clear()
        self.thread.join(timeout=1.0)
        self.cap.release()
