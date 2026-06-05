from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class FrameResult:
    seed: int
    detected: bool
    per_corner_err: Optional[np.ndarray]
    mean_err: Optional[float]

@dataclass
class LocalizationReport:
    n_frames: int
    n_detected: int
    n_success: int
    success_px: float
    mean_err_px: float
    median_err_px: float
    max_err_px: float
    per_frame: List[FrameResult]

    def detect_rate(self) -> float:
        return self.n_detected / max(self.n_frames, 1)

    def success_rate(self) -> float:
        return self.n_success / max(self.n_frames, 1)

    def summary(self) -> str:
        return f'n={self.n_frames}  detect={self.detect_rate():.1%}  success(<={self.success_px:g}px)={self.success_rate():.1%}  mean_err={self.mean_err_px:.2f}px  median_err={self.median_err_px:.2f}px  max_err={self.max_err_px:.2f}px'

def per_corner_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape != (4, 2) or gt.shape != (4, 2):
        raise ValueError('expected (4,2) corner arrays')
    return np.linalg.norm(pred - gt, axis=1)

def summarize(results: List[FrameResult], success_px: float) -> LocalizationReport:
    n = len(results)
    detected = [r for r in results if r.detected and r.mean_err is not None]
    errs = np.array([r.mean_err for r in detected], dtype=np.float64)
    n_success = int((errs <= success_px).sum()) if errs.size else 0
    return LocalizationReport(n_frames=n, n_detected=len(detected), n_success=n_success, success_px=success_px, mean_err_px=float(errs.mean()) if errs.size else float('nan'), median_err_px=float(np.median(errs)) if errs.size else float('nan'), max_err_px=float(errs.max()) if errs.size else float('nan'), per_frame=results)
