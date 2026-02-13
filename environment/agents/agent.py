import numpy as np
from utils.vec_utils import *

class Agent:
    def __init__(self, agent_id, pos, direction, seed):
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)

        self.pos = np.array(pos) # 3d
        self.norm_speed = 0.0
        self.direction = direction

    def move(self, dt):
        self.pos = self.pos + (self.direction * self.norm_speed * self.params.max_speed) * float(dt)

        if self.params.is_planar:
            self.pos[2] = 0.0
        else:
            self.pos[2] = max(self.pos[2], 0.0)

    def apply_control(self, direction, norm_speed, dt):
        direction = np.asarray(direction, dtype=float)

        if self.params.is_planar:
            direction[2] = 0.0

        self.direction = unit(direction)
        self.norm_speed = np.clip(norm_speed, 0.0, 1.0)

    def reflect_bounds(self):
        if self.pos[0] < 0.0:
            self.pos[0] = -self.pos[0]
            self.direction[0] *= -1.0
        elif self.pos[0] > self.x_bound:
            self.pos[0] = 2.0 * self.x_bound - self.pos[0]
            self.direction[0] *= -1.0

        if self.pos[1] < 0.0:
            self.pos[1] = -self.pos[1]
            self.direction[1] *= -1.0
        elif self.pos[1] > self.y_bound:
            self.pos[1] = 2.0 * self.y_bound - self.pos[1]
            self.direction[1] *= -1.0

    def to_dict(self):
        vx, vy, vz = self.direction * self.norm_speed * self.params.max_speed
        return {
            "agent_id": self.agent_id,
            "type": self.params.name,

            "x": self.pos[0],
            "y": self.pos[1],
            "z": self.pos[2],

            "vx": vx,
            "vy": vy,
            "vz": vz,

            "speed": self.norm_speed * self.params.max_speed,
        }

