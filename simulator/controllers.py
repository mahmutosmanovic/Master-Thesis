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

class BoundedRandomWalk(Controller):
    def __init__(
        self,
        speed: float = 1.0,
        change_prob: float = 0.15,
        max_change_angle_rad: float = np.deg2rad(25.0),
        seed: int = 0,
    ):
        self.speed = speed
        self.change_prob = change_prob
        self.max_change_angle = float(max_change_angle_rad)
        self.rng = np.random.default_rng(seed)
        self._dir: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._dir = None

    def act(self, obs) -> np.ndarray:
        if self._dir is None:
            self._dir = unit(self.rng.normal(size=3))
            return self.speed * self._dir

        if self.rng.random() < self.change_prob:
            new_dir = unit(self._dir + 0.5 * unit(self.rng.normal(size=3)))

            cosang = float(np.clip(np.dot(self._dir, new_dir), -1.0, 1.0))
            ang = float(np.arccos(cosang))
            if ang > self.max_change_angle:
                alpha = self.max_change_angle / (ang + 1e-9)
                new_dir = unit((1 - alpha) * self._dir + alpha * new_dir)

            self._dir = new_dir

        return self.speed * self._dir