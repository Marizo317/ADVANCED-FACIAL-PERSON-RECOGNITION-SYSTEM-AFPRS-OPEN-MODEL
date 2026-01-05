# vision/app/core/llm_client.py
import httpx
import asyncio
from threading import Thread, Event
from queue import Queue
from dataclasses import dataclass
from typing import Optional, Dict
import time
from config import CFG

@dataclass
class DescRequest:
    track_id: int
    age: Optional[int]
    gender: Optional[str]
    emotion: Optional[str]
    gaze: Optional[str]
    upper_color: Optional[str]
    lower_color: Optional[str]
    position: Optional[str]
    depth: Optional[str]

class LLMClient:
    def __init__(self):
        self.queue = Queue(maxsize=5)
        self.cache: Dict[int, tuple] = {}
        self.running = Event()
        self.running.set()
        
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[LLM_CLIENT] Connected to {CFG.LLM_URL}")
    
    def _loop(self):
        while self.running.is_set():
            try:
                req = self.queue.get(timeout=0.1)
                desc = self._generate(req)
                self.cache[req.track_id] = (desc, time.time())
            except:
                continue
    
    def _generate(self, req: DescRequest) -> str:
        try:
            with httpx.Client(timeout=CFG.LLM_TIMEOUT) as client:
                resp = client.post(f"{CFG.LLM_URL}/generate", json={
                    "age": req.age,
                    "gender": req.gender,
                    "emotion": req.emotion,
                    "gaze": req.gaze,
                    "upper_color": req.upper_color,
                    "lower_color": req.lower_color,
                    "position": req.position,
                    "depth": req.depth
                })
                if resp.status_code == 200:
                    return resp.json().get("description", "")
        except Exception as e:
            print(f"[LLM_CLIENT] Error: {e}")
        return ""
    
    def request(self, track):
        if track.id in self.cache:
            _, ts = self.cache[track.id]
            if time.time() - ts < CFG.DESC_INTERVAL:
                return
        
        req = DescRequest(
            track_id=track.id,
            age=track.age,
            gender=track.gender,
            emotion=track.emotion,
            gaze=track.head_pose[0] if track.head_pose else None,
            upper_color=track.clothing_upper,
            lower_color=track.clothing_lower,
            position=track.position_zone,
            depth=None
        )
        
        if not self.queue.full():
            self.queue.put(req)
    
    def get(self, track_id: int) -> Optional[str]:
        if track_id in self.cache:
            return self.cache[track_id][0]
        return None
    
    def stop(self):
        self.running.clear()
