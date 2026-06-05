from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
CELL_SIZE: int = 50
GRID_SIZE: int = CELL_SIZE * 9

def order_corners(pts: np.ndarray) -> np.ndarray:
    if pts.shape != (4, 2):
        raise ValueError(f'expected (4,2) corner array, got {pts.shape!r}')
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered

def compute_homography(src_corners: np.ndarray) -> np.ndarray:
    src = order_corners(src_corners)
    dst = np.array([[0, 0], [GRID_SIZE - 1, 0], [GRID_SIZE - 1, GRID_SIZE - 1], [0, GRID_SIZE - 1]], dtype=np.float32)
    (H, _) = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError('findHomography returned None for the given corners')
    return H

def rectify_grid(image: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = compute_homography(corners)
    warped = cv2.warpPerspective(image, H, (GRID_SIZE, GRID_SIZE))
    return (warped, H)

def split_cells(grid_image: np.ndarray) -> np.ndarray:
    if grid_image.shape[0] != GRID_SIZE or grid_image.shape[1] != GRID_SIZE:
        raise ValueError(f'expected a {GRID_SIZE}x{GRID_SIZE} grid, got {grid_image.shape!r}')
    cells = []
    for r in range(9):
        row = []
        for c in range(9):
            (y0, x0) = (r * CELL_SIZE, c * CELL_SIZE)
            row.append(grid_image[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE])
        cells.append(row)
    return np.array(cells)
