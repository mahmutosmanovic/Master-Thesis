import numpy as np
from environment.settings import *
from utils.vec_utils import *
from abc import ABC, abstractmethod

class Behaviour(ABC):
    def __init__(self, seed): # Seed using np seed sequencer
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, obs, params, dt):
        raise NotImplementedError
    
    # for when we have behaviour state, (memory etc.)
    def update(self, obs): 
        pass

    def reset(self):
        pass

class RandomWalk(Behaviour):
    def __init__(self, seed):
        super().__init__(seed)

    def act(self, obs, params, dt):
        cur_dir = unit(obs["direction"])
        cur_speed = float(obs["speed"])

        # Small random steering (simple: add noise in 3D and renormalize)
        desired_dir = cur_dir + self.rng.normal(0.0, params.turn_noise, size=3)
        desired_dir = unit(desired_dir)

        # Random-ish speed around mid range
        target_speed = 0.5 * params.max_speed
        desired_speed = target_speed + self.rng.normal(0.0, 0.2 * params.max_speed)
        desired_speed = float(np.clip(desired_speed, 0.0, params.max_speed))

        return desired_dir, desired_speed


class PathFollow(Behaviour):
    def __init__(self, path, seed):
        super().__init__(seed)
        self.path = path
        self.s = 0.0
        self.s_speed = 1.0
        self.Kp = 0.01

    def act(self, obs, params, dt):
        pos = np.asarray(obs["pos"], dtype=float)
        cur_dir = unit(obs["direction"])
        cur_speed = float(obs["speed"])

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

        desired_speed = float(0.7 * params.max_speed)

        # Advance path parameter based on current tangential speed
        v = cur_dir * cur_speed
        v_tan = max(0.0, float(np.dot(v, t_hat)))
        self.s += self.s_speed * (v_tan / (dp_norm + 1e-12)) * float(dt)

        return desired_dir, desired_speed

class POI(Behaviour):
    def __init__(self, pois, seed):
        super().__init__(seed)
        self.pois = [np.asarray(p, dtype=float) for p in pois]
        self.poi_idx = None

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

        if dist < POI_REACHED_EPS and POI_SWITCH_ON_REACH and len(self.pois) > 1:
            self.poi_idx = self._select_poi_idx()
            target = self.pois[self.poi_idx]
            to_target = target - pos
            dist = float(np.linalg.norm(to_target))

        desired_dir = unit(to_target)

        # Add a bit of directional noise (simple)
        noise = self.rng.normal(0.0, params.turn_noise, size=3) * NOISE_SCALE
        desired_dir = unit(desired_dir + noise)

        desired_speed = float(params.max_speed)

        return desired_dir, desired_speed