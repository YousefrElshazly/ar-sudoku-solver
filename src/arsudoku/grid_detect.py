from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

@dataclass(frozen=True)
class GridDetectConfig:
    approx_epsilon_frac: float = 0.02
    min_area_frac: float = 0.05
    require_convex: bool = True
DEFAULT_CFG = GridDetectConfig()

def find_grid_quad(binary: np.ndarray, cfg: GridDetectConfig=DEFAULT_CFG) -> Optional[np.ndarray]:
    if binary.ndim != 2:
        raise ValueError('expected a single-channel binary image')
    (h, w) = binary.shape
    image_area = float(h * w)
    min_area = cfg.min_area_frac * image_area
    (contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: Optional[np.ndarray] = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, cfg.approx_epsilon_frac * peri, closed=True)
        if approx.shape[0] != 4:
            continue
        if cfg.require_convex and (not cv2.isContourConvex(approx)):
            continue
        if area > best_area:
            best_area = area
            best_quad = approx
    if best_quad is None:
        return None
    return best_quad.reshape(4, 2).astype(np.float32)
