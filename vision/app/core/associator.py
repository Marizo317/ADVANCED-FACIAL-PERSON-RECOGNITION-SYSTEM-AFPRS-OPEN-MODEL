# vision/app/core/associator.py
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from config import CFG

@dataclass
class Association:
    person_idx: int
    face_idx: Optional[int]
    person: any
    face: any
    confidence: float

class Associator:
    def __init__(self):
        self.iou_thresh = 0.3
        self.vert_thresh = 0.4
    
    def associate(self, persons: List, faces: List) -> List[Association]:
        if not persons:
            return [Association(-1, i, None, f, f.confidence) for i, f in enumerate(faces)]
        if not faces:
            return [Association(i, None, p, None, p.confidence) for i, p in enumerate(persons)]
        
        # Score matrix
        scores = np.zeros((len(persons), len(faces)))
        for i, p in enumerate(persons):
            for j, f in enumerate(faces):
                scores[i, j] = self._score(p, f)
        
        # Greedy matching
        assocs = []
        used_p, used_f = set(), set()
        
        while True:
            if scores.max() < self.iou_thresh:
                break
            pi, fi = np.unravel_index(scores.argmax(), scores.shape)
            assocs.append(Association(pi, fi, persons[pi], faces[fi], scores[pi, fi]))
            used_p.add(pi)
            used_f.add(fi)
            scores[pi, :] = 0
            scores[:, fi] = 0
        
        # Unmatched persons
        for i, p in enumerate(persons):
            if i not in used_p:
                assocs.append(Association(i, None, p, None, p.confidence))
        
        # Unmatched faces
        for j, f in enumerate(faces):
            if j not in used_f:
                assocs.append(Association(-1, j, None, f, f.confidence * 0.5))
        
        return assocs
    
    def _score(self, p, f) -> float:
        px1, py1, px2, py2 = p.bbox
        fx1, fy1, fx2, fy2 = f.bbox
        
        ph = py2 - py1
        fc_y = (fy1 + fy2) / 2
        fc_x = (fx1 + fx2) / 2
        
        # Face must be in upper part
        if fc_y > py1 + ph * self.vert_thresh:
            return 0.0
        if fc_x < px1 or fc_x > px2:
            return 0.0
        
        # IoU with head region
        head = [px1, py1, px2, py1 + int(ph * 0.3)]
        iou = self._iou(f.bbox, head)
        return min(1.0, iou * (1 + p.confidence * 0.2 + f.confidence * 0.2))
    
    def _iou(self, b1, b2) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0
