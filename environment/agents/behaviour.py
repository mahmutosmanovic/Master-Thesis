import numpy as np
from functools import singledispatch
from dataclasses import dataclass
from environment.utils.vec_utils import unit
from environment.immutables import BehaviourState

# Base
@dataclass(frozen=True)
class BehaviourConfig:
    pass

@dataclass(frozen=True)
class CRWConfig(BehaviourConfig):
    persistence: float  = 0.9
    turn_sigma: float   = 0.15
    target_speed: float = 0.7
    speed_sigma: float  = 0.03
    speed_smooth: float = 0.2
    bias_gain: float    = 0.0

@singledispatch
def make_behaviour(cfg):
    raise TypeError(f"No behaviours for config type: {type(cfg).__name__}")

class Behaviour:
    def step_direction(self, cur_dir, cfg:CRWConfig, rng, bias_vec = np.zeros(3, dtype=float)):
        cur_dir = unit(cur_dir)
        persistence = np.clip(cfg.persistence, 0.0, 1.0)
        turn_sigma = np.max([0.0, cfg.turn_sigma])

        # turning noise
        noise = rng.normal(0.0, turn_sigma, size=3)
        bias = unit(bias_vec) * cfg.bias_gain

        # correlated update
        desired = persistence * cur_dir + noise + bias
        return unit(desired)

    def step_speed(self, cur_norm_speed, rng, cfg:CRWConfig):
        speed_smooth = np.clip(cfg.speed_smooth, 0.0, 1.0)
        speed_sigma = np.max([0.0, cfg.speed_sigma])

        norm_speed = (1.0 - speed_smooth) * cur_norm_speed + speed_smooth * cfg.target_speed + rng.normal(0.0, speed_sigma)
        return np.clip(norm_speed, 0.0, 1.0)
    
    def act(self, obs, rng, dt):
        raise NotImplementedError

    def get_state(self) -> str:
        return BehaviourState.EXPLORE

# Implementations

@make_behaviour.register
def _(cfg: CRWConfig):
    return CRW(cfg)

class CRW(Behaviour):
    def __init__(self, cfg):
        self.cfg = cfg

    def act(self, obs, rng, dt):
        desired_dir = self.step_direction(cur_dir=obs["direction"], rng=rng, cfg=self.cfg)
        desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], rng=rng, cfg=self.cfg)
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
        bias_gain = 0.0
        )
    
    time_to_leave: float = 10

@make_behaviour.register
def _(cfg: ExploreExploitConfig):
    return ExploreExploit(cfg)

class ExploreExploit(Behaviour):
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = BehaviourState.EXPLORE
        self.time_since_encounter = 0.0

    def update_state(self, obs, dt):
        encounter = obs["encounter"]
        if encounter:
            self.state = BehaviourState.EXPLOIT
            self.time_since_encounter = 0.0
            return
        
        if self.state == BehaviourState.EXPLOIT:
            self.time_since_encounter += dt
            if self.time_since_encounter > self.cfg.time_to_leave:
                self.state = BehaviourState.EXPLORE

    def act(self, obs, rng, dt):
        self.update_state(obs, dt)
        # Aplly movement
        match self.state:
            case BehaviourState.EXPLORE:
                desired_dir = self.step_direction(cur_dir=obs["direction"], rng=rng, cfg=self.cfg.explore_cfg)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], rng=rng, cfg=self.cfg.explore_cfg)
            case BehaviourState.EXPLOIT:
                desired_dir = self.step_direction(cur_dir=obs["direction"], rng=rng, cfg=self.cfg.exploit_cfg)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], rng=rng, cfg=self.cfg.exploit_cfg)
            case _:
                raise NotImplementedError
            
        return desired_dir, desired_norm_speed
    
    def get_state(self) -> str:
        return self.state

@dataclass(frozen=True)
class TraplineConfig(BehaviourConfig):
    travel_cfg: CRWConfig = CRWConfig(
        persistence = 0.90,
        turn_sigma = 0.15,
        target_speed = 0.7,
        speed_sigma = 0.03,
        speed_smooth = 0.2,
        bias_gain = 0.3
    )
    patch_cfg: CRWConfig = CRWConfig(
        persistence = 0.9,
        turn_sigma = 0.3,
        target_speed = 0.3,
        speed_sigma = 0.03,
        speed_smooth = 0.2,
        bias_gain = 0.0
    )

    arrive_dist: float = 5.0
    patch_time: float = 15.0
    top_k_pois: int = 4

@make_behaviour.register
def _(cfg: TraplineConfig):
    return Trapline(cfg)

class Trapline(Behaviour):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.state = BehaviourState.EXPLORE
        self.route = None
        self.route_index = 0
        self.time_in_patch = 0.0

    def _init_route(self, pois, rng):
        if self.cfg.top_k_pois is None:
            self.route = pois
        else:
            self.route = pois[:self.cfg.top_k_pois]

        rng.shuffle(self.route)
        self.route_index = 0

    def update_state(self, obs, rng, dt):
        pois = obs.get("pois", [])
        pos = obs["pos"]

        if self.route is None:
            self._init_route(pois, rng)

        target = self.route[self.route_index]
        bias_vec = np.array([target[0] - pos[0], target[1] - pos[1], 0])
        dist = np.linalg.norm(bias_vec)

        if self.state == BehaviourState.EXPLORE:
            if dist < self.cfg.arrive_dist:
                self.state = BehaviourState.EXPLOIT
                self.time_in_patch = 0.0

        elif self.state == BehaviourState.EXPLOIT:
            self.time_in_patch += dt
            if self.time_in_patch > self.cfg.patch_time:
                self.route_index = (self.route_index + 1) % len(self.route)
                self.state = BehaviourState.EXPLORE
        
        return bias_vec
    
    def act(self, obs, rng, dt):
        bias_vec = self.update_state(obs, rng, dt)
        # Aplly movement
        match self.state:
            case BehaviourState.EXPLORE:
                desired_dir = self.step_direction(cur_dir=obs["direction"], rng=rng, cfg=self.cfg.travel_cfg, bias_vec=bias_vec)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], rng=rng, cfg=self.cfg.travel_cfg)
            case BehaviourState.EXPLOIT:
                desired_dir = self.step_direction(cur_dir=obs["direction"], rng=rng, cfg=self.cfg.patch_cfg)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], rng=rng, cfg=self.cfg.patch_cfg)
            case _:
                raise NotImplementedError
            
        return desired_dir, desired_norm_speed
        

    def get_state(self):
        return self.state