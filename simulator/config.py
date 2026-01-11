from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class WorldConfig:
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (100.0, 100.0, 30.0)
    dt: float = 1
    max_steps: int = 200

    delta_disturb: float = 12.0   # spatial decay


def default_config() -> WorldConfig:
    return WorldConfig()
