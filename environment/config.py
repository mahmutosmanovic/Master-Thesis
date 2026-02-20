from dataclasses import dataclass, field
from typing import Sequence, Optional
from environment.agents.behaviour import BehaviourConfig

@dataclass
class EnvConfig:
   # simulation
   dt: float = 0.2      # (seconds)
   max_t: float = 200.0 # (seconds)

   # map
   map_size: float = 1000.0    # (meters)
   map_altitude: float = 100.0 # (meters)

   # animals
   force_bounds: bool = False # restrict animal movement to map bounds ([0, map_size] x [0, map_size])
   # each entry: {params: AnimalParams, count: int, behaviour: BehaviourConfig}
   animals: Sequence[dict] = field(default_factory=list)

   # animal resources
   p_wavelenght: float = 200.0, # (meters) wavelenght of major resource noise
   p_reduction: float = 0.2,    # Reduction on raw encounter probability
   p_scale: float = 0.4,        # Scaling of reduced probability
   sample_res: float = 5.0,     # (meters per sample) Sample resolution for poi generation
   min_poi_p: float = 1e-2,     # minimum value for a local maxima to be considered a poi
   kernel_size: float = 250.0,  # (meters) kernel size for poi generation (local maxima)

   # drones
   drone_target_order: str = "random", # Drone target order ("round_robin", "random"), selects how to assign drones to animals
   # each entry: {params: DroneParams, count: int, spawn_range: [int, int]}
   drones: Sequence[dict] = field(default_factory=list)
    
   # observation and reward
   distance_scale: float = 5.0
   alignment_scale: float = 1.0
   disturbance_scale: float = 2.5
   control_scale:float = 0.5

@dataclass
class AnimalConfig:
   # metadata
   name: str = "standard_animal" # animal type name

   # geometry
   is_planar: bool = True # restricted to z=0?

   # movement
   max_speed: float = 12.0 # (m/s) maximum speed

   # flight/avoid
   avoidance_threshold: float = 0.75 # disturbance threshold to initiate avoiding behaviour
   flight_threshold: float = 1       # disturbance threshold to initiate fleeing behaviour

@dataclass
class DroneConfig:
   # metadata
   name: str = "standard_drone" # drone type name

   # geometry
   is_planar: bool = False # restricted to z=0?

   # movement
   max_speed: float    = 12.0 # (m/s) maximum speed
   max_view_yaw: float = 2    # (radians) maximum view rotation speed 

   # camera parameters
   camera_pitch: float = -30 # (degrees) camera pitch (0 -> horizontal, -90 -> straight down, 90 -> straight up)
   hfov: float         = 90  # (degrees) horizontal field of view
   vfov: float         = 56  # (degrees) vertical field of view
   near_plane: float   = 1   # (meters) frustum near plane distance
   far_plane: float    = 200 # (meters) frustum far plane distance
   max_targets: int    = 1   # Number of slots in observation, if lower than number of animals includes nearest max_targets