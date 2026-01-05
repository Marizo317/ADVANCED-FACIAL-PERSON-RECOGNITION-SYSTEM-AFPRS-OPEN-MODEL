# vision/app/core/tracker.py
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time
from config import CFG

@dataclass
class Track:
    id: int
    person_bbox: Optional[List[int]] = None
    face_bbox: Optional[List[int]] = None
    person_conf: float = 0.0
    face_conf: float = 0.0
    keypoints: Optional[np.ndarray] = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=CFG.TRAJECTORY_LEN))
    
    # Analysis data
    landmarks_468: Optional[np.ndarray] = None
    iris_left: Optional[np.ndarray] = None
    iris_right: Optional[np.ndarray] = None
    head_pose: Optional[Tuple[float, float, float]] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    clothing_upper: Optional[str] = None
    clothing_lower: Optional[str] = None
    position_zone: Optional[str] = None
    description: Optional[str] = None
    
    hits: int = 1
    age_frames: int = 0
    time_since_update: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def update_trajectory(self):
        bbox = self.person_bbox or self.face_bbox
        if bbox:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            self.trajectory.append((cx, cy))
    
    def get_conf(self) -> float:
        if self.person_conf > 0 and self.face_conf > 0:
            return (self.person_conf + self.face_conf) / 2
        return max(self.person_conf, self.face_conf)

class Tracker:
    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
    
    def update(self, assocs: List) -> List[Track]:
        for t in self.tracks.values():
            t.age_frames += 1
            t.time_since_update += 1
        
        if not assocs:
            self._cleanup()
            return list(self.tracks.values())
        
        if not self.tracks:
            for a in assocs:
                self._create(a)
            return list(self.tracks.values())
        
        # Match
        matched, unm_tracks, unm_assocs = self._match(assocs)
        
        for tid, aidx in matched:
            self._update(tid, assocs[aidx])
        
        for aidx in unm_assocs:
            self._create(assocs[aidx])
        
        self._cleanup()
        return list(self.tracks.values())
    
    def _create(self, a):
        t = Track(id=self.next_id)
        if a.person:
            t.person_bbox = a.person.bbox
            t.person_conf = a.person.confidence
            t.keypoints = a.person.keypoints
        if a.face:
            t.face_bbox = a.face.bbox
            t.face_conf = a.face.confidence
        t.update_trajectory()
        self.tracks[self.next_id] = t
        self.next_id += 1
    
    def _update(self, tid: int, a):
        t = self.tracks[tid]
        if a.person:
            t.person_bbox = a.person.bbox
            t.person_conf = a.person.confidence
            t.keypoints = a.person.keypoints
        if a.face:
            t.face_bbox = a.face.bbox
            t.face_conf = a.face.confidence
        t.hits += 1
        t.time_since_update = 0
        t.last_seen = time.time()
        t.update_trajectory()
    
    def _match(self, assocs):
        tids = list(self.tracks.keys())
        t_boxes = [self.tracks[tid].person_bbox or self.tracks[tid].face_bbox or [0,0,0,0] for tid in tids]
        a_boxes = [a.person.bbox if a.person else (a.face.bbox if a.face else [0,0,0,0]) for a in assocs]
        
        iou_mat = np.zeros((len(tids), len(assocs)))
        for i, tb in enumerate(t_boxes):
            for j, ab in enumerate(a_boxes):
                iou_mat[i, j] = self._iou(tb, ab)
        
        matched = []
        unm_t = set(range(len(tids)))
        unm_a = set(range(len(assocs)))
        
        while iou_mat.max() >= CFG.MATCH_THRESH:
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            matched.append((tids[i], j))
            unm_t.discard(i)
            unm_a.discard(j)
            iou_mat[i, :] = 0
            iou_mat[:, j] = 0
        
        return matched, list(unm_t), list(unm_a)
    
    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0
    
    def _cleanup(self):
        dead = [tid for tid, t in self.tracks.items() if t.time_since_update > CFG.TRACK_BUFFER]
        for tid in dead:
            del self.tracks[tid]
