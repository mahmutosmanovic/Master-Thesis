import numpy as np
from functools import singledispatch
from dataclasses import dataclass
from utils.vec_utils import *

class CRWKernel: # Base movement for all behaviours
    def __init__(self, rng):
        self.rng = rng

    def step_direction(self, cur_dir, persistence, turn_sigma, bias_vec = np.zeros(3, dtype=float), bias_gain = 1.0):
        cur_dir = unit(cur_dir)
        persistence = np.clip(persistence, 0.0, 1.0)
        turn_sigma = np.max([0.0, turn_sigma])

        # turning noise
        noise = self.rng.normal(0.0, turn_sigma, size=3)
        bias = unit(bias_vec) * bias_gain

        # correlated update
        desired = persistence * cur_dir + noise + bias
        return unit(desired)

    def step_speed(self, cur_norm_speed, target_norm_speed, speed_sigma = 0.0, smooth = 0.2):
        smooth = np.clip(smooth, 0.0, 1.0)
        speed_sigma = np.max([0.0, speed_sigma])

        norm_speed = (1.0 - smooth) * cur_norm_speed + smooth * target_norm_speed + self.rng.normal(0.0, speed_sigma)
        return np.clip(norm_speed, 0.0, 1.0)

# Base
@dataclass
class BehaviourConfig:
    pass

@singledispatch
def make_behaviour(cfg, seed):
    raise TypeError(f"No behaviours for config type: {type(cfg).__name__}")

class Behaviour:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.kernel = CRWKernel(self.rng)

    def act(self, obs, dt):
        raise NotImplementedError

    # for when we have behaviour state, (memory etc.)
    def update(self, obs):
        pass

    def reset(self):
        pass

    def get_state(self) -> str:
        return "base"

# Implementations

@dataclass
class CRWConfig(BehaviourConfig):
    persistence: float  = 0.9
    turn_sigma: float   = 0.25
    target_speed: float = 0.7
    speed_sigma: float  = 0.03
    speed_smooth: float = 0.2
    bias_gain: float    = 0.0

@make_behaviour.register
def _(cfg: CRWConfig, seed):
    return CRW(cfg, seed)

class CRW(Behaviour):
    def __init__(self, cfg, seed):
        super().__init__(seed)
        self.cfg = cfg

    def act(self, obs, dt):
        desired_dir = self.kernel.step_direction(
            cur_dir=obs["direction"],
            persistence=self.cfg.persistence,
            turn_sigma=self.cfg.turn_sigma,
            bias_gain=self.cfg.bias_gain,
        )
        desired_norm_speed = self.kernel.step_speed(
            cur_norm_speed=obs["norm_speed"],
            target_norm_speed=self.cfg.target_speed,
            speed_sigma=self.cfg.speed_sigma,
            smooth=self.cfg.speed_smooth,
        )
        return desired_dir, desired_norm_speed

# Old implementations

class RandomWalk(Behaviour):
    def __init__(self, seed):
        super().__init__(seed)

    def act(self, obs, params, dt):
        cur_dir = unit(obs["direction"])
        cur_norm_speed = float(obs["norm_speed"])

        # Small random steering (simple: add noise in 3D and renormalize)
        desired_dir = cur_dir + self.rng.normal(0.0, params.turn_noise, size=3)
        desired_dir = unit(desired_dir)

        # Random-ish norm_speed around mid range
        target_norm_speed = 0.5
        desired_norm_speed = target_norm_speed + self.rng.normal(0.0, 0.2)
        desired_norm_speed = float(np.clip(desired_norm_speed, 0.0, 1))

        return desired_dir, desired_norm_speed

class PathFollow(Behaviour):
    def __init__(self, path, seed):
        super().__init__(seed)
        self.path = path
        self.s = 0.0
        self.s_norm_speed = 1.0
        self.Kp = 0.01

    def act(self, obs, params, dt):
        pos = np.asarray(obs["pos"], dtype=float)
        cur_dir = unit(obs["direction"])
        cur_norm_speed = float(obs["norm_speed"])

        # Project onto path and get path frame
        self.s = self.path.project_s(pos, self.s, iters=2)
        p = self.path.position(self.s)
        dp = self.path.tangent(self.s)
        dp_norm = float(np.linalg.norm(dp))
        t_hat = unit(dp) if dp_norm > 1e-8 else np.array([1.0, 0.0, 0.0])

        # Cross-track correction
        to_path = p - pos
        e_ct = to_path - np.dot(to_path, t_hat) * t_hat
        desired_dir = unit(t_hat + self.Kp * e_ct)

        # Optional exploration noise
        if self.rng.random() < params.epsilon:
            desired_dir = unit(desired_dir + self.rng.normal(0.0, params.turn_noise * 0.3, size=3))

        desired_norm_speed = 0.7

        # Advance path parameter based on current tangential norm_speed
        v = cur_dir * cur_norm_speed
        v_tan = max(0.0, float(np.dot(v, t_hat)))
        self.s += self.s_norm_speed * (v_tan / (dp_norm + 1e-12)) * float(dt)

        return desired_dir, desired_norm_speed

class POI(Behaviour):
    def __init__(self, pois, seed, poi_reached_eps=3.0):
        super().__init__(seed)
        self.pois = [np.asarray(p, dtype=float) for p in pois]
        self.poi_idx = None
        self.poi_reached_eps = poi_reached_eps

    def _select_poi_idx(self):
        if self.poi_idx is None:
            return int(self.rng.integers(0, len(self.pois)))
        choices = list(range(len(self.pois)))
        choices.remove(self.poi_idx)
        return int(self.rng.choice(choices))

    def act(self, obs, params, dt):
        if not self.pois:
            raise ValueError("No points of interest!")

        if self.poi_idx is None:
            self.poi_idx = self._select_poi_idx()

        pos = np.asarray(obs["pos"], dtype=float)

        target = self.pois[self.poi_idx]
        to_target = target - pos

        if params.is_planar:
            to_target[2] = 0.0

        dist = float(np.linalg.norm(to_target))

        if dist < self.poi_reached_eps and len(self.pois) > 1:
            self.poi_idx = self._select_poi_idx()
            target = self.pois[self.poi_idx]
            to_target = target - pos
            dist = float(np.linalg.norm(to_target))

        desired_dir = unit(to_target)

        # Add a bit of directional noise (simple)
        noise = self.rng.normal(0.0, params.turn_noise, size=3)
        desired_dir = unit(desired_dir + noise)

        desired_norm_speed = 0.7

        return desired_dir, desired_norm_speed