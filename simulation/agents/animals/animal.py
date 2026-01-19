import numpy as np
from ..agent import Agent

class Animal(Agent):
    def __init__(self, pos, behaviour):
        super().__init__(pos)
        self.behaviour = behaviour

    # Policy dispatch
    def policy(self, obs):
        return self.behaviour.act(obs, self._get_params())
    
    def _get_params(self):
        params = {"epsilon": self.epsilon,
                  "max_speed": self.max_speed,
                  "max_turn": self.max_turn,
                  "max_accel": self.max_accel,
                  "vision": self.vision,
                  "turn_noise": self.turn_noise,
                  "memory_decay": self.memory_decay,}
        return params

    def __repr__(self):
        x, y, z = self.pos
        return f"{type(self).__name__}([{round(x,1)}, {round(y,1)}, {round(z,1)}], mode={type(self.behaviour).__name__})"

# Species
class Eagle(Animal):
    def __init__(self, pos, behaviour):
        super().__init__(pos, behaviour)
        # path following
        self.epsilon = 0.3

        # movement limits
        self.max_speed = 30.0
        self.max_turn  = 8.0
        self.max_accel = 8.0

        # perception & cognition
        self.vision = 250.0
        self.turn_noise = 2.5
        self.memory_decay = 0.995

class Jackal(Animal):
    def __init__(self, pos, behaviour):
        super().__init__(pos, behaviour)
        # path following
        self.epsilon = 0.6

        # movement limits
        self.max_speed = 6.0
        self.max_turn  = 4.0
        self.max_accel = 4.0

        # perception & cognition
        self.vision = 100.0
        self.turn_noise = 1.5
        self.memory_decay = 0.98

class Pigeon(Animal):
    def __init__(self, pos, behaviour):
        super().__init__(pos, behaviour)
        # path following
        self.epsilon = 0.8

        # movement limits
        self.max_speed = 15.0
        self.max_turn  = 16.0
        self.max_accel = 6.0

        # perception & cognition
        self.vision = 80.0
        self.turn_noise = 3.25
        self.memory_decay = 0.97
        