from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class PreprocessConfig:
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    blur_ksize: int = 5
    blur_sigma: float = 0.0
    adaptive_block: int = 19
    adaptive_C: int = 7

    def __post_init__(self) -> None:
        if self.adaptive_block % 2 == 0 or self.adaptive_block < 3:
            raise ValueError('adaptive_block must be an odd integer >= 3')
        if self.blur_ksize % 2 == 0 or self.blur_ksize < 1:
            raise ValueError('blur_ksize must be a positive odd integer')
DEFAULT_CFG = PreprocessConfig()

def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError(f'unsupported image shape {image.shape!r}')

def preprocess(image: np.ndarray, cfg: PreprocessConfig=DEFAULT_CFG, return_stages: bool=False):
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (cfg.blur_ksize, cfg.blur_ksize), cfg.blur_sigma)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, cfg.adaptive_block, cfg.adaptive_C)
    if return_stages:
        stages = {'gray': gray, 'equalized': equalized, 'blurred': blurred, 'binary': binary}
        return (binary, stages)
    return binary
