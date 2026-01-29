import numpy as np
from utils.vec_utils import *

class Agent:
    def __init__(self, agent_id, pos, seed, mode=None):
        self.agent_id = agent_id

        self.rng = np.random.default_rng(seed)

        self.pos = np.array(pos) # 3d
        self.speed = 0.0
        self.direction = self.random_direction()

        self.mode = mode

    def l2_norm(self, v):
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return v / n

    def random_direction(self):
        v = self.rng.normal(size=3)
        return self.l2_norm(v)

    def move(self, dt):
        self.pos = self.pos + (self.direction * self.speed) * float(dt)

        if self.params.is_planar:
            self.pos[2] = 0.0
        else:
            self.pos[2] = max(self.pos[2], 0.0)

    def update(self, action, dt):
        direction, speed = action
        self.apply_control(direction, speed, dt)
        self.move(dt)

    def apply_control(self, direction, speed, dt):
        direction = np.asarray(direction, dtype=float)

        if self.params.is_planar:
            direction[2] = 0.0

        d_current = self.direction
        norm = np.linalg.norm(direction)

        if norm < 1e-8:
            self.direction = d_current
        else:
            d_desired = direction / norm

            # Enforce max turn, may not be neccesary after fit, or, estimate from data ???
            theta = angle_between(d_current, d_desired)
            theta_max = self.params.max_turn * float(dt)

            if theta > theta_max: 
                t = theta_max / theta
                self.direction = slerp(d_current, d_desired, t)
            else:
                self.direction = d_desired

        self.speed = float(speed)

    def to_dict(self):
        vx, vy, vz = self.direction * self.speed
        return {
            "agent_id": self.agent_id,
            "species": self.params.name,
            "mode": self.mode,

            "x": self.pos[0],
            "y": self.pos[1],
            "z": self.pos[2],

            "vx": vx,
            "vy": vy,
            "vz": vz,

            "speed": self.speed,
        }

