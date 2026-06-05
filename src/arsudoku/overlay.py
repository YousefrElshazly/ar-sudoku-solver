from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
from .rectify import CELL_SIZE, GRID_SIZE
INK_BGR: Tuple[int, int, int] = (0, 90, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_THICK = 2

def render_overlay_canvas(recognized: np.ndarray, solved: np.ndarray, color: Tuple[int, int, int]=INK_BGR) -> Tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for r in range(9):
        for c in range(9):
            if recognized[r, c] != 0:
                continue
            d = int(solved[r, c])
            if d == 0:
                continue
            text = str(d)
            ((tw, th), _) = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
            cx = c * CELL_SIZE + (CELL_SIZE - tw) // 2
            cy = r * CELL_SIZE + (CELL_SIZE + th) // 2
            cv2.putText(canvas, text, (cx, cy), FONT, FONT_SCALE, color, FONT_THICK, cv2.LINE_AA)
            cv2.rectangle(mask, (c * CELL_SIZE, r * CELL_SIZE), ((c + 1) * CELL_SIZE - 1, (r + 1) * CELL_SIZE - 1), 255, -1)
    return (canvas, mask)

def draw_overlay(frame: np.ndarray, H: np.ndarray, recognized: np.ndarray, solved: np.ndarray) -> np.ndarray:
    (h, w) = frame.shape[:2]
    (canvas, _) = render_overlay_canvas(recognized, solved)
    digit_mask = (canvas.sum(axis=2) > 0).astype(np.uint8) * 255
    H_inv = np.linalg.inv(H)
    warped_digits = cv2.warpPerspective(canvas, H_inv, (w, h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(digit_mask, H_inv, (w, h), flags=cv2.INTER_LINEAR)
    alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
    out = frame.astype(np.float32) * (1.0 - alpha) + warped_digits.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
