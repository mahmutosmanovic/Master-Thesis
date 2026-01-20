import numpy as np
from ..agent import Agent

from dataclasses import dataclass
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

class Animal(Agent):
    def __init__(self, pos, params, behaviour, seed):
        super().__init__(pos, seed)
        self.params = params

        self.max_speed = params.max_speed
        self.max_turn = params.max_turn
        self.max_accel = params.max_accel
        self.turn_noise = params.turn_noise
        self.epsilon = params.epsilon
        self.is_planar = params.is_planar
    
        self.behaviour = behaviour

    def load_params(self, param_path):
        pass

    # Policy dispatch
    def policy(self, obs, dt):
        return self.behaviour.act(obs, self.params, dt)

    def __repr__(self):
        x, y, z = self.pos
        return f"{self.params.name}([{round(x,1)}, {round(y,1)}, {round(z,1)}], behaviour={type(self.behaviour).__name__}, id={self.agent_id})"

def jackal_params():
    return AnimalParams(
        name="jackal",
        is_planar=True,
        max_speed=12.0,
        max_turn=4.0,
        max_accel=4.0,
        turn_noise=1.5,
        epsilon=0.05,
    )

def eagle_params():
   return AnimalParams(
       name="eagle",
       is_planar=False,
       max_speed=30.0,
       max_turn=8.0,
       max_accel=8.0,
       turn_noise=2.5,
       epsilon=0.03,
   )

def pigeon_params():
    return AnimalParams(
        name="pigeon",
        is_planar = False,
        max_speed = 15.0,
        max_turn  = 16.0,
        max_accel = 6.0,
        turn_noise = 3.25,
        epsilon = 0.8,
    )

        