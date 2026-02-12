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
        bias_gain = 0.0
        )
    
    time_to_leave: float = 10

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
        self.time_since_encounter = 0.0

    def update_state(self, obs, dt):
        encounter = self.rng.random() < obs["p_resource"]
        if encounter:
            print("encounter")
            self.state = self.STATE_EXPLOIT
            self.time_since_encounter = 0.0
            return
        
        if self.state == self.STATE_EXPLOIT:
            self.time_since_encounter += dt
            if self.time_since_encounter > self.cfg.time_to_leave:
                print("exploring")
                self.state = self.STATE_EXPLORE

    def act(self, obs, dt):
        self.update_state(obs, dt)
        # Aplly movement
        match self.state:
            case ExploreExploit.STATE_EXPLORE:
                desired_dir = self.step_direction(cur_dir=obs["direction"], cfg=self.cfg.explore_cfg)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], cfg=self.cfg.explore_cfg)
            case ExploreExploit.STATE_EXPLOIT:
                desired_dir = self.step_direction(cur_dir=obs["direction"], cfg=self.cfg.exploit_cfg)
                desired_norm_speed = self.step_speed(cur_norm_speed=obs["norm_speed"], cfg=self.cfg.exploit_cfg)
            case _:
                raise NotImplementedError
            
        return desired_dir, desired_norm_speed
    
    def get_state(self) -> str:
        return self.state