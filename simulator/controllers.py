from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from utils import unit


class Controller(ABC):
    def reset(self) -> None:
        pass

    @abstractmethod
    def act(self, obs) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs, action_vel: np.ndarray) -> None:
        pass


class RandomWalk(Controller):
    def __init__(self, speed: float = 1.0, change_prob: float = 0.15, seed: int = 0):
        self.speed = speed
        self.change_prob = change_prob
        self.rng = np.random.default_rng(seed)
        self._dir: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._dir = None

    def act(self, obs) -> np.ndarray:
        if self._dir is None or self.rng.random() < self.change_prob:
            self._dir = unit(self.rng.normal(size=3))
        return self.speed * self._dir
