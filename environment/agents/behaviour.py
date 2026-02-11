import numpy as np
from functools import singledispatch
from dataclasses import dataclass
from utils.vec_utils import *

# Base
@dataclass(frozen=True)
class BehaviourConfig:
    pass

@dataclass(frozen=True)
class CRWConfig(BehaviourConfig):
    persistence: float  = 0.9
    turn_sigma: float   = 0.25
    target_speed: float = 0.7
    speed_sigma: float  = 0.03
    speed_smooth: float = 0.2
    bias_gain: float    = 0.0

@singledispatch
def make_behaviour(cfg, seed):
    raise TypeError(f"No behaviours for config type: {type(cfg).__name__}")

class Behaviour:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def step_direction(self, cur_dir, cfg:CRWConfig, bias_vec = np.zeros(3, dtype=float)):
        cur_dir = unit(cur_dir)
        persistence = np.clip(cfg.persistence, 0.0, 1.0)
        turn_sigma = np.max([0.0, cfg.turn_sigma])

        # turning noise
        noise = self.rng.normal(0.0, turn_sigma, size=3)
        bias = unit(bias_vec) * cfg.bias_gain

        # correlated update
        desired = persistence * cur_dir + noise + bias
        return unit(desired)

    def step_speed(self, cur_norm_speed, cfg:CRWConfig):
        speed_smooth = np.clip(cfg.speed_smooth, 0.0, 1.0)
        speed_sigma = np.max([0.0, cfg.speed_sigma])

        norm_speed = (1.0 - speed_smooth) * cur_norm_speed + speed_smooth * cfg.target_speed + self.rng.normal(0.0, speed_sigma)
        return np.clip(norm_speed, 0.0, 1.0)
    
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

@make_behaviour.register
def _(cfg: CRWConfig, seed):
    return CRW(cfg, seed)

class CRW(Behaviour):
    def __init__(self, cfg, seed):
        super().__init__(seed)
        self.cfg = cfg

    def act(self, obs, dt):
        desired_dir = self.step_direction(cur_dir=obs["direction"], cfg=self.cfg)
        desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], cfg=self.cfg)
        return desired_dir, desired_norm_speed

@dataclass(frozen=True)
class ExploreExploitConfig(BehaviourConfig):
    explore_cfg: CRWConfig = CRWConfig(
        persistence = 0.9,
        turn_sigma = 0.15,
        target_speed = 0.7,
        speed_sigma = 0.03,
        speed_smooth = 0.2,
        bias_gain = 0.0
        )
    exploit_cfg: CRWConfig = CRWConfig(
        persistence = 0.9,
        turn_sigma = 0.3,
        target_speed = 0.3,
        speed_sigma = 0.03,
        speed_smooth = 0.2,
        bias_gain = 0.1
        )
    p_explore: float = 0.01
    p_exploit: float = 0.008

@make_behaviour.register
def _(cfg: ExploreExploitConfig, seed):
    return ExploreExploit(cfg, seed)

class ExploreExploit(Behaviour):
    STATE_EXPLORE = "explore"
    STATE_EXPLOIT = "exploit"
    def __init__(self, cfg, seed):
        super().__init__(seed)
        self.cfg = cfg
        self.state = ExploreExploit.STATE_EXPLORE
        self.exploit_point = np.zeros(3)

    def update_state(self, obs, dt):
        chance = self.rng.random()
        match self.state:
            case ExploreExploit.STATE_EXPLORE:  # maybe set state from observation? would require some sort of reward, perlin noise reward field?
                if chance < self.cfg.p_exploit:
                    self.exploit_point = obs["pos"]
                    self.state = ExploreExploit.STATE_EXPLOIT
            case ExploreExploit.STATE_EXPLOIT:
                if chance < self.cfg.p_explore:
                    self.state = ExploreExploit.STATE_EXPLORE
            case _:
                raise NotImplementedError

    def act(self, obs, dt):
        self.update_state(obs, dt)
        # Aplly movement
        match self.state:
            case ExploreExploit.STATE_EXPLORE:
                desired_dir = self.step_direction(cur_dir=obs["direction"], cfg=self.cfg.explore_cfg)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], cfg=self.cfg.explore_cfg)
            case ExploreExploit.STATE_EXPLOIT:
                bias_dir = unit(self.exploit_point - obs["pos"])
                desired_dir = self.step_direction(cur_dir=obs["direction"], cfg=self.cfg.exploit_cfg, bias_vec=bias_dir)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], cfg=self.cfg.exploit_cfg)
            case _:
                raise NotImplementedError
            
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