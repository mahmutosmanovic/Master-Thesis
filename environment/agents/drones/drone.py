from ..agent import Agent
from dataclasses import dataclass

class Drone(Agent):
    def __init__(self, pos, params, seed):
        super().__init__(pos, seed)
        self.params = params
        self.pos = pos

@dataclass
class DroneParams:
   # metadata
   name: str

   # geometry
   is_planar: bool

   # movement
   max_speed: float
   max_turn: float
   max_accel: float

   # noise / behavior
   turn_noise: float
   epsilon: float

def drone_params():
    return DroneParams(
        name="jackal",
        is_planar=True,
        max_speed=12.0,
        max_turn=4.0,
        max_accel=4.0,
        turn_noise=1.5,
        epsilon=0.8,
    )