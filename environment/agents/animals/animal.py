import numpy as np
from ..agent import Agent
from dataclasses import dataclass
from utils.vec_utils import *

class Animal(Agent):
    def __init__(self, agent_id, pos, params, behaviour, seed, mode=None):
        super().__init__(agent_id, pos, seed, mode)
        self.params = params
        self.behaviour = behaviour
        self.state = "base"

    # Behavior
    def policy(self, obs, dt):
        direction, speed = self.behaviour.act(obs, self.params, dt) # always act, to maintain determinism
        disturbance = np.sum([drone["val"] for drone in obs["disturbance_info"].values()])

        if disturbance > 0.95: # Flee !!! hardcoded atm
            mean_disturbance_dir = self.calc_mean_disturbance_dir(obs)
            self.state = "flee"
            return mean_disturbance_dir, self.params.max_speed
        elif disturbance > 0.6: # Avoid ! hardcoded atm
            mean_disturbance_dir = self.calc_mean_disturbance_dir(obs)

            w = (disturbance - 0.4) / 0.4
            blended_dir = (1 - w) * direction + w * mean_disturbance_dir
            self.state = "avoid"
            return unit(blended_dir), speed
        else: # Base
            self.state = "base"
            return direction, speed
    
    def calc_mean_disturbance_dir(self, obs):
        drone_directions = [drone["dir"] for drone in obs["disturbance_info"].values()]
        mean_disturbance_dir = np.mean(drone_directions, axis=0)
        if self.params.is_planar:
            mean_disturbance_dir[2] == 0
        return -unit(mean_disturbance_dir)

    def __repr__(self):
        x, y, z = self.pos
        return f"{self.params.name}([{round(x,1)}, {round(y,1)}, {round(z,1)}], behaviour={type(self.behaviour).__name__}, id={self.agent_id})"

@dataclass
class AnimalParams:
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

def jackal_params():
    return AnimalParams(
        name="jackal",
        is_planar=True,
        max_speed=12.0,
        max_turn=4.0,
        max_accel=4.0,
        turn_noise=0.4,
        epsilon=1,
    )

def eagle_params():
   return AnimalParams(
       name="eagle",
       is_planar=False,
       max_speed=30.0,
       max_turn=8.0,
       max_accel=8.0,
       turn_noise=0.4,
       epsilon=0.03,
   )

def pigeon_params():
    return AnimalParams(
        name="pigeon",
        is_planar = False,
        max_speed = 15.0,
        max_turn  = 16.0,
        max_accel = 6.0,
        turn_noise = 0.6,
        epsilon = 0.8,
    )

        