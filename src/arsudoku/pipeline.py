from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from . import recognize, solve
from .grid_detect import find_grid_quad
from .overlay import draw_overlay
from .preprocess import preprocess as _preprocess
from .rectify import order_corners, rectify_grid, split_cells

@dataclass
class PipelineResult:
    success: bool
    detected_corners: Optional[np.ndarray]
    homography: Optional[np.ndarray]
    rectified: Optional[np.ndarray]
    recognized: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    solved: Optional[np.ndarray]
    overlay: Optional[np.ndarray]
    note: str = ''

def _retry_low_confidence(digits: np.ndarray, logits: np.ndarray, confidence: np.ndarray, k_max: int=5) -> Optional[np.ndarray]:
    n_cells = 9 * 9
    flat_conf = confidence.ravel().copy()
    flat_digits = digits.ravel().copy()
    candidate_idx = [i for i in np.argsort(flat_conf) if flat_digits[i] != 0]
    for k in range(1, min(k_max, len(candidate_idx)) + 1):
        worst = candidate_idx[:k]
        from itertools import product
        alt_options = []
        for i in worst:
            (r, c) = divmod(i, 9)
            ranked = np.argsort(-logits[r, c])
            alt_options.append([int(d) for d in ranked[:3]])
        for combo in product(*alt_options):
            test = digits.copy()
            for (i, d) in zip(worst, combo):
                (r, c) = divmod(i, 9)
                test[r, c] = d
            sol = solve.solve(test)
            if sol is not None:
                return sol
    return None

def run_pipeline(image: np.ndarray, model: recognize.DigitCNN, device=None) -> PipelineResult:
    binary = _preprocess(image)
    corners = find_grid_quad(binary)
    if corners is None:
        return PipelineResult(False, None, None, None, None, None, None, None, 'no grid found')
    corners = order_corners(corners)
    (warped, H) = rectify_grid(image, corners)
    cells = split_cells(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    rec = recognize.predict_grid(model, cells, device=device)
    solved = solve.solve(rec.digits)
    if solved is None:
        solved = _retry_low_confidence(rec.digits, rec.logits, rec.confidence)
        if solved is None:
            return PipelineResult(False, corners, H, warped, rec.digits, rec.confidence, None, None, note='recognition produced an unsolvable board (after retry)')
    overlay = draw_overlay(image, H, rec.digits, solved)
    return PipelineResult(True, corners, H, warped, rec.digits, rec.confidence, solved, overlay)
