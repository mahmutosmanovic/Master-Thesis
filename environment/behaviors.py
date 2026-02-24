from .vec import Vector
from .immutables import MovementDim, BehaviorState
from dataclasses import dataclass

@dataclass(frozen=True)
class CRW_CFG:
    persistence: float  = 0.9
    turn_sigma: float   = 0.15
    target_speed: float = 10   # (m/s)
    speed_sigma: float  = 1    # (m/s)
    speed_smooth: float = 0.2
    bias_gain: float    = 0.0

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

@dataclass(frozen=True)
class POI_CFG:
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
        bias_gain = 0.5
        )

    arrive_dist: float = 10.0
    time_to_leave: float = 15.0

BEHAVIOR_REGISTRY = {}

class BehaviorBase:
    cfg_type = None

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

class ExploreExploit(BehaviorBase):
    cfg_type = EE_CFG
    def __init__(self, cfg):
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
        self.time_since_encounter = 0.0
        
class PointOfInterest(BehaviorBase):
    cfg_type = POI_CFG
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = BehaviorState.EXPLORE
        self.time_since_arrival = 0.0
        self.target = None

    def fn(self, animal, rng, dt):
        # update state
        bias_vec = None
        if self.state == BehaviorState.EXPLORE:
            if self.target == None:
                pois = animal.resource_map.get_pois()
                poi = rng.choice(pois[0:1])   # choose from top candidate(s)
                self.target = Vector(poi[0], poi[1], 0) # pois are sorted based on probability of resource, could also be used for target selection

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
        self.time_since_arrival = 0.0
