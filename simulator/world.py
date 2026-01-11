from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from config import WorldConfig
from agent import Agent, AgentObs
from utils import clip_speed, clip_to_bounds


class World:
    def __init__(self, cfg: WorldConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.bounds_min = np.array(cfg.bounds_min, dtype=float)
        self.bounds_max = np.array(cfg.bounds_max, dtype=float)
        self.t: int = 0
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        for a in self.agents:
            a.state.pos = self.rng.uniform(a.params.bounds_min, a.params.bounds_max)
            a.state.vel = np.zeros(3, dtype=float)
            a.reset()

    def _calc_disturbance(self):
        drone_positions = [a.state.pos for a in self.agents if a.params.agent_type == "drone"]
        drone_positions = np.array(drone_positions, dtype=float) if len(drone_positions) else None

        disturb = np.zeros(len(self.agents), dtype=float)
        avoid_vecs = np.zeros((len(self.agents), 3), dtype=float)

        if drone_positions is not None:
            for i, a in enumerate(self.agents):
                tot = 0.0
                av = np.zeros(3, dtype=float)
                for dp in drone_positions:
                    dvec = a.state.pos - dp
                    d = float(np.linalg.norm(dvec)) + 1e-9
                    w = float(np.exp(-d / self.cfg.delta_disturb))
                    tot += w
                    av += w * (dvec / d)
                disturb[i] = tot
                avoid_vecs[i] = av
        
        return disturb, avoid_vecs

    def step(self, external_actions: Optional[Dict[str, np.ndarray]] = None) -> None:
        external_actions = external_actions or {}

        disturb, avoid_vecs = self._calc_disturbance()

        for i, a in enumerate(self.agents):
            obs = AgentObs(
                pos=a.state.pos.copy(),
                vel=a.state.vel.copy(),
                t=self.t,
                bounds_min=a.params.bounds_min,
                bounds_max=a.params.bounds_max,
                extras={
                    "disturb": float(disturb[i]),
                    "avoid_vec": avoid_vecs[i].copy(),
                    "rng": self.rng,
                },
            )

            if a.name in external_actions:
                v_cmd = np.asarray(external_actions[a.name], dtype=float)
            elif a.controller is not None:
                v_cmd = np.asarray(a.controller.act(obs), dtype=float)
            else:
                print(f"no action or controller was provided for agent {a.name}")
                raise ValueError

            v_cmd = clip_speed(v_cmd, a.params.max_speed)
            a.state.vel = v_cmd
            a.state.pos = clip_to_bounds(
                a.state.pos + v_cmd * self.cfg.dt,
                a.params.bounds_min,
                a.params.bounds_max,
            )

            if a.controller is not None and hasattr(a.controller, "update"):
                a.controller.update(obs, v_cmd)

        self.t += 1
