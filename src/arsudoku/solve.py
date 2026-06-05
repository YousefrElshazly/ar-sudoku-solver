from __future__ import annotations
from typing import Optional
import numpy as np

def _legal_candidates(board: np.ndarray, r: int, c: int) -> list[int]:
    row = set(board[r, :].tolist())
    col = set(board[:, c].tolist())
    (br, bc) = (r // 3 * 3, c // 3 * 3)
    box = set(board[br:br + 3, bc:bc + 3].ravel().tolist())
    used = row | col | box
    return [d for d in range(1, 10) if d not in used]

def _next_cell(board: np.ndarray) -> Optional[tuple[int, int, list[int]]]:
    best: Optional[tuple[int, int, list[int]]] = None
    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                continue
            cands = _legal_candidates(board, r, c)
            if not cands:
                return (r, c, [])
            if best is None or len(cands) < len(best[2]):
                best = (r, c, cands)
                if len(cands) == 1:
                    return best
    return best

def is_valid(board: np.ndarray) -> bool:
    for i in range(9):
        row = board[i, :]
        col = board[:, i]
        for arr in (row, col):
            vals = arr[arr != 0]
            if len(set(vals.tolist())) != len(vals):
                return False
    for br in range(3):
        for bc in range(3):
            box = board[br * 3:br * 3 + 3, bc * 3:bc * 3 + 3].ravel()
            vals = box[box != 0]
            if len(set(vals.tolist())) != len(vals):
                return False
    return True

def solve(board: np.ndarray, max_nodes: int=200000) -> Optional[np.ndarray]:
    if board.shape != (9, 9):
        raise ValueError(f'expected (9, 9) board, got {board.shape!r}')
    if not is_valid(board):
        return None
    work = board.astype(np.int32).copy()
    nodes = [0]

    def recurse() -> bool:
        nxt = _next_cell(work)
        if nxt is None:
            return True
        (r, c, cands) = nxt
        if not cands:
            return False
        for d in cands:
            nodes[0] += 1
            if nodes[0] > max_nodes:
                return False
            work[r, c] = d
            if recurse():
                return True
            work[r, c] = 0
        return False
    return work if recurse() else None
