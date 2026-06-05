from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise ImportError('Pillow is required for synth image generation') from exc
CANVAS = 600
GRID_PX = 450
MARGIN = (CANVAS - GRID_PX) // 2

@dataclass
class SynthConfig:
    seed: int = 0
    tilt_px: int = 90
    lighting_strength: float = 0.55
    noise_sigma: float = 12.0
    paper_tone: Tuple[int, int, int] = (242, 240, 232)
    ink_tone: Tuple[int, int, int] = (28, 28, 32)
    fill_ratio: float = 0.45
    thin_line_px: int = 2
    thick_line_px: int = 4

def _make_puzzle_digits(rng: np.random.Generator, fill_ratio: float) -> np.ndarray:
    grid = np.zeros((9, 9), dtype=np.int32)
    mask = rng.random((9, 9)) < fill_ratio
    digits = rng.integers(1, 10, size=(9, 9))
    grid[mask] = digits[mask]
    return grid

def _make_valid_puzzle(rng: np.random.Generator) -> np.ndarray:
    from .puzzles import random_puzzle
    (puzzle, _) = random_puzzle(rng)
    return puzzle

def _render_clean_grid(digits: np.ndarray, cfg: SynthConfig, font: Optional[ImageFont.ImageFont]=None) -> np.ndarray:
    img = Image.new('RGB', (CANVAS, CANVAS), cfg.paper_tone)
    draw = ImageDraw.Draw(img)
    cell = GRID_PX // 9
    (x0, y0) = (MARGIN, MARGIN)
    for i in range(10):
        w = cfg.thick_line_px if i % 3 == 0 else cfg.thin_line_px
        x = x0 + i * cell
        draw.line([(x, y0), (x, y0 + GRID_PX)], fill=cfg.ink_tone, width=w)
        y = y0 + i * cell
        draw.line([(x0, y), (x0 + GRID_PX, y)], fill=cfg.ink_tone, width=w)
    if font is None:
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 32)
        except OSError:
            font = ImageFont.load_default()
    for r in range(9):
        for c in range(9):
            d = int(digits[r, c])
            if d == 0:
                continue
            cx = x0 + c * cell + cell // 2
            cy = y0 + r * cell + cell // 2
            text = str(d)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                (tw, th) = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                draw.text((cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]), text, fill=cfg.ink_tone, font=font)
            except Exception:
                draw.text((cx - 6, cy - 8), text, fill=cfg.ink_tone, font=font)
    return np.array(img)

def _grid_corners() -> np.ndarray:
    return np.array([[MARGIN, MARGIN], [MARGIN + GRID_PX - 1, MARGIN], [MARGIN + GRID_PX - 1, MARGIN + GRID_PX - 1], [MARGIN, MARGIN + GRID_PX - 1]], dtype=np.float32)

def _random_tilt_corners(rng: np.random.Generator, tilt_px: int) -> np.ndarray:
    base = _grid_corners()
    jitter = rng.uniform(-tilt_px, tilt_px, size=base.shape).astype(np.float32)
    return base + jitter

def _apply_lighting_gradient(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    (h, w) = image.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    (ys, xs) = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    ramp = np.cos(angle) * xs + np.sin(angle) * ys
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-09)
    ramp = 1.0 - strength + strength * ramp
    out = image.astype(np.float32) * ramp[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)

def _add_noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

@dataclass
class SynthSample:
    image: np.ndarray
    corners: np.ndarray
    digits: np.ndarray
    seed: int = 0
    extras: dict = field(default_factory=dict)

def generate_sample(seed: int, cfg: SynthConfig=SynthConfig(), valid_puzzle: bool=False) -> SynthSample:
    rng = np.random.default_rng(seed)
    digits = _make_valid_puzzle(rng) if valid_puzzle else _make_puzzle_digits(rng, cfg.fill_ratio)
    clean_rgb = _render_clean_grid(digits, cfg)
    src = _grid_corners()
    dst = _random_tilt_corners(rng, cfg.tilt_px)
    (H, _) = cv2.findHomography(src, dst, method=0)
    warped_rgb = cv2.warpPerspective(clean_rgb, H, (CANVAS, CANVAS), borderMode=cv2.BORDER_CONSTANT, borderValue=cfg.paper_tone)
    lit = _apply_lighting_gradient(warped_rgb, rng, cfg.lighting_strength)
    noisy = _add_noise(lit, rng, cfg.noise_sigma)
    bgr = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
    corners_ordered = _order_tl_tr_br_bl(dst)
    return SynthSample(image=bgr, corners=corners_ordered, digits=digits, seed=seed, extras={'homography': H})

def _order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out

def generate_dataset(n: int, cfg: SynthConfig=SynthConfig(), base_seed: int=0) -> List[SynthSample]:
    return [generate_sample(base_seed + i, cfg) for i in range(n)]
