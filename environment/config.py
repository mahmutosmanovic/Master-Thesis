from dataclasses import dataclass, field
from typing import Sequence, Optional

@dataclass
class EnvConfig:
    # simulation
    dt: float = 0.2
    max_t: float = 200.0

    # map
    map_width: float = 200.0
    map_height: float = 200.0
    map_altitude: float = 100.0

    # POIs
    poi_count: int = 0
    poi_points: Optional[Sequence[tuple]] = None

    # animals
    # each entry: {params: AnimalParams, count: int, mode: str}
    animals: Sequence[dict] = field(default_factory=list)

    # drones
    # each entry: {params: DroneParams, count: int, sensor: str}
    drones: Sequence[dict] = field(default_factory=list)
    
    # reward
    penalty_scale: float = 2.5
    reward_scale: float = 5.0

@dataclass
class AnimalParams:
   # metadata
   name: str

   # geometry
   is_planar: bool

   # movement
   max_speed: float
   max_turn: float

   # noise / behavior
   turn_noise: float
   epsilon: float
   avoidance_threshold: float
   flight_threshold: float

@dataclass
class DroneParams:
   # metadata
   name: str

   # geometry
   is_planar: bool

   # movement
   max_speed: float
   max_turn: float
   max_view_yaw: float

   camera_pitch: float