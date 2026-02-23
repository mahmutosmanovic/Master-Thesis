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