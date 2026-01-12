from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class AgentParams:
    max_speed: float
    bounds_min: np.ndarray  # (3,)
    bounds_max: np.ndarray  # (3,)
    agent_type: str         # e.g. "animal" or "drone"


@dataclass
class AgentState:
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)


@dataclass
class AgentObs:
    pos: np.ndarray
    vel: np.ndarray
    t: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    extras: Dict[str, Any]


class Agent:
    def __init__(self, name: str, params: AgentParams, controller: Optional[object] = None):
        self.name = name
        self.params = params
        self.controller = controller
        self.state = AgentState(pos=np.zeros(3, dtype=float), vel=np.zeros(3, dtype=float))

    def reset(self) -> None:
        if self.controller is not None and hasattr(self.controller, "reset"):
            self.controller.reset()
