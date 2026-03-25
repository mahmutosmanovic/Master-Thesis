import numpy as np
import pandas as pd
from .vec import Vector
from pathlib import Path
from .immutables import MovementDim, BehaviorState
from dataclasses import dataclass
from box import Box

BEHAVIOR_REGISTRY = {}

class BehaviorBase:
    cfg_type = None
    can_flee = True
    handles_spawn = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.cfg_type is not None:
            BEHAVIOR_REGISTRY[cls.cfg_type] = cls

# helpers
def step_direction(animal, cfg, rng, bias_vec=None):
        # turning noise
        noise = Vector(*rng.normal(0.0, cfg.turn_sigma, size=3))
        if bias_vec is None:
            bias = Vector()
        else:
            bias = bias_vec.getter().unit().scale(cfg.bias_gain)

        # correlated update
        animal.vel_dir = animal.vel_dir.unit().scale(cfg.persistence).add(noise).add(bias).unit()
        if animal.movement_dim == MovementDim.TWO_D:
            animal.vel_dir.z = 0.0
            animal.vel_dir.unit()

def step_speed(animal, cfg, rng):
    animal.vel_speed = (1.0 - cfg.speed_smooth) * animal.vel_speed + cfg.speed_smooth * cfg.target_speed + rng.normal(0.0, cfg.speed_sigma)

@dataclass(frozen=True)
class CRW_CFG:
    persistence: float  = 0.9
    turn_sigma: float   = 0.15
    target_speed: float = 10   # (m/s)
    speed_sigma: float  = 1    # (m/s)
    speed_smooth: float = 0.2
    bias_gain: float    = 0.0

#behaviour classes
class CorrelatedRandomWalk(BehaviorBase):
    cfg_type = CRW_CFG
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = BehaviorState.EXPLORE

    def fn(self, animal, rng, dt):
        step_direction(animal, self.cfg, rng)
        step_speed(animal, self.cfg, rng)
    
    def reset(self):
        pass

@dataclass(frozen=True)
class EE_CFG:
    explore_cfg: CRW_CFG = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.15,
        target_speed = 10,
        speed_sigma = 1,
        speed_smooth = 0.2,
        bias_gain = 0.0
        )
    exploit_cfg: CRW_CFG = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.3,
        target_speed = 4,
        speed_sigma = 1,
        speed_smooth = 0.2,
        bias_gain = 0.0
        )
    
    time_to_leave: float = 10

class ExploreExploit(BehaviorBase):
    cfg_type = EE_CFG
    def __init__(self, cfg):
        if isinstance(cfg.explore_cfg, dict):
            cfg = EE_CFG(
                explore_cfg=CRW_CFG(**cfg.explore_cfg),
                exploit_cfg=CRW_CFG(**cfg.exploit_cfg),
                time_to_leave=cfg.time_to_leave,
            )
        self.cfg = cfg
        self.state = BehaviorState.EXPLORE
        self.time_since_encounter = 0.0

    def fn(self, animal, rng, dt):
        # update state
        encounter, _ = animal.resource_map.is_encounter(animal.pos, rng)

        if encounter:
            self.state = BehaviorState.EXPLOIT
            self.time_since_encounter = 0.0
        else:
            if self.state == BehaviorState.EXPLOIT:
                self.time_since_encounter += dt
                if self.time_since_encounter > self.cfg.time_to_leave:
                    self.state = BehaviorState.EXPLORE

        # update vel and speed
        match self.state:
            case BehaviorState.EXPLORE:
                step_direction(animal, self.cfg.explore_cfg, rng)
                step_speed(animal, self.cfg.explore_cfg, rng)
            case BehaviorState.EXPLOIT:
                step_direction(animal, self.cfg.exploit_cfg, rng)
                step_speed(animal, self.cfg.exploit_cfg, rng)
            case _:
                raise NotImplementedError
    
    def reset(self):
        self.state = BehaviorState.EXPLORE
        self.time_since_encounter = 0.0

@dataclass(frozen=True)
class POI_CFG:
    explore_cfg: CRW_CFG = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.15,
        target_speed = 10,
        speed_sigma = 1,
        speed_smooth = 0.2,
        bias_gain = 0.5
        )
    exploit_cfg: CRW_CFG = CRW_CFG(
        persistence = 0.9,
        turn_sigma = 0.3,
        target_speed = 4,
        speed_sigma = 1,
        speed_smooth = 0.2,
        bias_gain = 0.0
        )

    arrive_dist: float = 10.0
    time_to_leave: float = 15.0

class PointOfInterest(BehaviorBase):
    cfg_type = POI_CFG
    def __init__(self, cfg):
        if isinstance(cfg.explore_cfg, dict):
            cfg = POI_CFG(
                explore_cfg=CRW_CFG(**cfg.explore_cfg),
                exploit_cfg=CRW_CFG(**cfg.exploit_cfg),
                time_to_leave=cfg.time_to_leave,
                arrive_dist=cfg.arrive_dist
            )
        self.cfg = cfg
        self.state = BehaviorState.EXPLORE
        self.time_since_arrival = 0.0
        self.target = None

    def choose_target(self, animal, rng, k=10):
        pois = animal.resource_map.get_pois()
        if len(pois) == 0:
            print("[Warning] POIS are empty")
            return

        d = np.hypot(pois[:, 0] - animal.pos.x, pois[:, 1] - animal.pos.y)
        nearest_idx = np.argsort(d)[:min(k, len(pois))]
        poi = pois[int(rng.choice(nearest_idx))]
        return Vector(float(poi[0]), float(poi[1]), 0.0)

    def fn(self, animal, rng, dt):
        # update state
        bias_vec = None
        if self.state == BehaviorState.EXPLORE:
            if self.target is None:
                self.target = self.choose_target(animal, rng)

            bias_vec = self.target.add(animal.pos.scale(-1)) # scale is safe, other functions can mutate state...
            dist = bias_vec.norm()

            if dist < self.cfg.arrive_dist:
                self.target = None
                self.state = BehaviorState.EXPLOIT
                self.time_since_arrival = 0.0
        elif self.state == BehaviorState.EXPLOIT:
            self.time_since_arrival += dt

            if self.time_since_arrival > self.cfg.time_to_leave:
                self.state = BehaviorState.EXPLORE
        
        # update vel and speed
        match self.state:
            case BehaviorState.EXPLORE:
                step_direction(animal, self.cfg.explore_cfg, rng, bias_vec=bias_vec)
                step_speed(animal, self.cfg.explore_cfg, rng)
            case BehaviorState.EXPLOIT:
                step_direction(animal, self.cfg.exploit_cfg, rng)
                step_speed(animal, self.cfg.exploit_cfg, rng)
            case _:
                raise NotImplementedError
    
    def reset(self):
        self.state = BehaviorState.EXPLORE
        self.time_since_arrival = 0.0
        self.target = None

@dataclass(frozen=True)
class LPOI_CFG:
    explore_cfg: CRW_CFG = CRW_CFG(
        persistence=0.9,
        turn_sigma=0.15,
        target_speed=10,
        speed_sigma=1,
        speed_smooth=0.2,
        bias_gain=0.5
    )
    exploit_cfg: CRW_CFG = CRW_CFG(
        persistence=0.9,
        turn_sigma=0.3,
        target_speed=4,
        speed_sigma=1,
        speed_smooth=0.2,
        bias_gain=0.0
    )

    arrive_dist: float = 10.0
    time_to_leave: float = 15.0

    epsilon: float = 0.1        # chance for random POI selection
    learning_rate: float = 0.3  # how fast POI values update
    disturbance_penalty: float = 1.0   # how much disturbance adds to the internal reward

class LearningPointOfInterest(BehaviorBase):
    cfg_type = LPOI_CFG

    def __init__(self, cfg):
        if isinstance(cfg.explore_cfg, dict):
            cfg = LPOI_CFG(
                explore_cfg=CRW_CFG(**cfg.explore_cfg),
                exploit_cfg=CRW_CFG(**cfg.exploit_cfg),
                time_to_leave=cfg.time_to_leave,
                arrive_dist=cfg.arrive_dist
            )
        self.cfg = cfg
        self.state = BehaviorState.EXPLORE
        self.time_since_arrival = 0.0
        self.target = None
        self.target_key = None

        self.poi_values = {}   # (x, y) -> learned value
        self.reward_sum = 0.0

    def choose_target(self, animal, rng, k=10):
        pois = np.asarray(animal.resource_map.get_pois(), dtype=float)
        if len(pois) == 0:
            print("[Warning] POIS are empty")
            return None, None

        # populate values for unseen POIs
        for poi in pois:
            key = (float(poi[0]), float(poi[1]))
            if key not in self.poi_values:
                self.poi_values[key] = 0.0

        # restrict choice to k nearest POIs
        dists = np.hypot(pois[:, 0] - animal.pos.x, pois[:, 1] - animal.pos.y)
        k = min(k, len(pois))
        nearest_idx = np.argsort(dists)[:k]
        nearest_pois = pois[nearest_idx]

        # epsilon-greedy within nearest set
        if rng.random() < self.cfg.epsilon:
            idx_local = int(rng.integers(0, len(nearest_pois)))
            poi = nearest_pois[idx_local]
        else:
            poi = max(
                nearest_pois,
                key=lambda p: self.poi_values[(float(p[0]), float(p[1]))]
            )

        key = (float(poi[0]), float(poi[1]))
        target = Vector(float(poi[0]), float(poi[1]), 0.0)
        return key, target

    def fn(self, animal, rng, dt):
        encounter, _ = animal.resource_map.is_encounter(animal.pos, rng)
        bias_vec = None

        if self.state == BehaviorState.EXPLORE:
            if self.target is None:
                self.target_key, self.target = self.choose_target(animal, rng)

            if self.target is not None:
                bias_vec = self.target.add(animal.pos.scale(-1))
                dist = bias_vec.norm()

                if dist < self.cfg.arrive_dist:
                    self.state = BehaviorState.EXPLOIT
                    self.time_since_arrival = 0.0
                    self.reward_sum = 0.0

        elif self.state == BehaviorState.EXPLOIT:
            self.time_since_arrival += dt
            self.reward_sum -= animal.disturbance * self.cfg.disturbance_penalty

            if encounter:
                self.reward_sum += 1.0

            if self.time_since_arrival > self.cfg.time_to_leave:
                old = self.poi_values[self.target_key]
                reward = self.reward_sum
                self.poi_values[self.target_key] = old + self.cfg.learning_rate * (reward - old)

                self.state = BehaviorState.EXPLORE
                self.time_since_arrival = 0.0
                self.target = None
                self.target_key = None
                self.reward_sum = 0.0

        match self.state:
            case BehaviorState.EXPLORE:
                step_direction(animal, self.cfg.explore_cfg, rng, bias_vec=bias_vec)
                step_speed(animal, self.cfg.explore_cfg, rng)
            case BehaviorState.EXPLOIT:
                step_direction(animal, self.cfg.exploit_cfg, rng)
                step_speed(animal, self.cfg.exploit_cfg, rng)
            case _:
                raise NotImplementedError

    def reset(self):
        self.state = BehaviorState.EXPLORE
        self.time_since_arrival = 0.0
        self.target = None
        self.target_key = None
        self.reward_sum = 0.0
        self.poi_values = {}

@dataclass(frozen=True)
class REPLAY_CFG:
    manifest_path: str
    selection: str = "cycle"
    zero_centered: bool = False

class Replay(BehaviorBase):
    cfg_type = REPLAY_CFG
    can_flee = False
    handles_spawn = True

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = BehaviorState.EXPLORE

        self.manifest_path = Path(cfg.manifest_path)
        self.base_dir = self.manifest_path.parent
        self.manifest = pd.read_parquet(self.manifest_path).reset_index(drop=True)

        self.current_idx = -1
        self.t = None
        self.x = None
        self.y = None
        self.elapsed = 0.0

    def _pick_idx(self, rng):
        if self.cfg.selection == "random":
            return int(rng.integers(0, len(self.manifest)))
        return (self.current_idx + 1) % len(self.manifest)

    def _load_segment(self, animal, rng):
        self.current_idx = self._pick_idx(rng)
        row = self.manifest.iloc[self.current_idx]

        path = self.base_dir / row["path"]
        with np.load(path, allow_pickle=False) as z:
            self.t = z["t"]
            self.x = z["x"]
            self.y = z["y"]
        
        if self.cfg.zero_centered:
            self.x = self.x - np.mean(self.x)
            self.y = self.y - np.mean(self.y)

        self.elapsed = 0.0
        animal.pos = Vector(self.x[0], self.y[0], 0.0)
        animal.vel_dir = Vector(1.0, 0.0, 0.0)
        animal.vel_speed = 0.0

    def _interp_xy(self, t_now):
        if t_now >= self.t[-1]:
            return self.x[-1], self.y[-1]

        j = np.searchsorted(self.t, t_now, side="right") - 1
        j = max(0, min(j, len(self.t) - 2))

        t0, t1 = self.t[j], self.t[j + 1]
        x0, x1 = self.x[j], self.x[j + 1]
        y0, y1 = self.y[j], self.y[j + 1]

        if t1 == t0:
            return x1, y1

        a = (t_now - t0) / (t1 - t0)
        x = x0 + a * (x1 - x0)
        y = y0 + a * (y1 - y0)
        return x, y

    def fn(self, animal, rng, dt):
        if self.elapsed >= self.t[-1]:
            animal.vel_dir = Vector(0.0, 0.0, 0.0)
            animal.vel_speed = 0.0
            return True

        t_next = min(self.elapsed + dt, self.t[-1])
        x_target, y_target = self._interp_xy(t_next)

        dx = x_target - animal.pos.x
        dy = y_target - animal.pos.y
        dist = np.sqrt(dx * dx + dy * dy)

        if dist == 0.0:
            animal.vel_dir = Vector(0.0, 0.0, 0.0)
            animal.vel_speed = 0.0
        else:
            animal.vel_dir = Vector(dx, dy, 0.0).unit()
            animal.vel_speed = dist / dt

        self.elapsed = t_next

    def reset(self, animal=None, rng=None):
        self.state = BehaviorState.EXPLORE
        self.t = None
        self.x = None
        self.y = None
        self.elapsed = 0.0

        if animal is not None and rng is not None:
            self._load_segment(animal, rng)