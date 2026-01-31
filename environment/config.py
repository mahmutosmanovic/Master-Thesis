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
