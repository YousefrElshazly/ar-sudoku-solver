from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError('PyTorch is required for arsudoku.recognize') from exc
CELL_PX = 50
INSET_PX = 6
N_CLASSES = 10

class DigitCNN(nn.Module):

    def __init__(self, n_classes: int=N_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

@dataclass
class RecognitionResult:
    digits: np.ndarray
    confidence: np.ndarray
    logits: np.ndarray

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def _to_gray(cell: np.ndarray) -> np.ndarray:
    if cell.ndim == 3:
        (b, g, r) = (cell[..., 0], cell[..., 1], cell[..., 2])
        return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)
    return cell.astype(np.float32)

def _inner_crop(cell: np.ndarray) -> np.ndarray:
    import cv2
    (h, w) = cell.shape[:2]
    inset = INSET_PX
    inner = cell[inset:h - inset, inset:w - inset]
    return cv2.resize(inner, (CELL_PX, CELL_PX), interpolation=cv2.INTER_AREA)

def _preprocess_cells(cells: np.ndarray) -> torch.Tensor:
    cropped = [_inner_crop(c) for c in cells]
    arr = np.stack([_to_gray(c) for c in cropped])
    arr = arr / 255.0
    arr = arr - arr.mean(axis=(1, 2), keepdims=True)
    return torch.from_numpy(arr).float().unsqueeze(1)

def predict_grid(model: DigitCNN, cells: np.ndarray, device: Optional[torch.device]=None) -> RecognitionResult:
    if cells.shape[0] != 9 or cells.shape[1] != 9:
        raise ValueError(f'expected (9, 9, 50, 50[, C]), got {cells.shape!r}')
    device = device or _device()
    flat = cells.reshape(81, *cells.shape[2:])
    x = _preprocess_cells(flat).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        (conf, pred) = probs.max(dim=1)
    digits = pred.cpu().numpy().reshape(9, 9).astype(np.int32)
    confidence = conf.cpu().numpy().reshape(9, 9).astype(np.float32)
    return RecognitionResult(digits=digits, confidence=confidence, logits=logits.cpu().numpy().reshape(9, 9, N_CLASSES))

def load_model(weights_path: Path, device: Optional[torch.device]=None) -> DigitCNN:
    device = device or _device()
    model = DigitCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model
