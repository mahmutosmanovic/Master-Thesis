from __future__ import annotations
import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def clip_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= max_speed:
        return v
    return (v / (n + 1e-9)) * max_speed


def clip_to_bounds(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return np.minimum(np.maximum(p, lo), hi)
