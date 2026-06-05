from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from .pipeline import run_pipeline
from .recognize import load_model
from .rectify import order_corners
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / 'results' / 'digit_cnn.pt'

def _draw_quad(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    out = image.copy()
    pts = order_corners(corners).astype(int)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=3)
    for ((x, y), name) in zip(pts, ['TL', 'TR', 'BR', 'BL']):
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(out, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out

def main(argv: list[str] | None=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('image', type=Path)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    args = p.parse_args(argv)
    if not args.image.exists():
        print(f'input not found: {args.image}', file=sys.stderr)
        return 2
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f'could not read image: {args.image}', file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f'weights not found at {args.weights}\nrun: python scripts/train_digit_cnn.py --n 600 --epochs 6', file=sys.stderr)
        return 2
    model = load_model(args.weights)
    result = run_pipeline(image, model)
    if not result.success:
        print(f'pipeline failed: {result.note}', file=sys.stderr)
        return 1
    out_dir = args.out if args.out is not None else args.image.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    cv2.imwrite(str(out_dir / f'{stem}_overlay.png'), _draw_quad(image, result.detected_corners))
    cv2.imwrite(str(out_dir / f'{stem}_warped.png'), result.rectified)
    cv2.imwrite(str(out_dir / f'{stem}_ar.png'), result.overlay)
    print(f'wrote {stem}_overlay.png, {stem}_warped.png, {stem}_ar.png to {out_dir}')
    print(f'recognized:\n{result.recognized}')
    print(f'solved:\n{result.solved}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
