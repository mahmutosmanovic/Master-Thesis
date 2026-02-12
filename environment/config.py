from dataclasses import dataclass, field
from typing import Sequence, Optional
from environment.agents.behaviour import BehaviourConfig

@dataclass
class EnvConfig:
    # simulation
    dt: float = 0.2
    max_t: float = 200.0

    # map
    map_width: float = 200.0
    map_height: float = 200.0
    map_altitude: float = 100.0

    # animals
    # each entry: {params: AnimalParams, count: int, mode: str}
    animals: Sequence[dict] = field(default_factory=list)

    # animal resources
    resource_frequency: float = 0.006
    resource_scale: float = 0.5
    resource_abundance: float = 0.4

    # drones
    # each entry: {params: DroneParams, count: int, sensor: str}
    drones: Sequence[dict] = field(default_factory=list)
    
    # observation and reward
    distance_scale: float = 5.0
    alignment_scale: float = 1.0
    disturbance_scale: float = 2.5
    control_scale:float = 0.5

@dataclass
class AnimalParams:
   # metadata
   name: str

   # geometry
   is_planar: bool

   # movement
   max_speed: float
   max_turn: float

   # flight/avoid
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