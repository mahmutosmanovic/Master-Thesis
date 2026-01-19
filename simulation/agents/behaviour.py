import numpy as np
from simulation.settings import *
from abc import ABC, abstractmethod

class Behaviour(ABC):
    def __init__(self, seed): # Seed using np seed sequencer
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, obs, params):
        raise NotImplementedError
    
    # for when we have behaviour state, (memory etc.)
    def update(self, obs): 
        pass

    def reset(self):
        pass

class RandomWalk(Behaviour):
    def __init__(self, seed):
        super().__init__(seed)
    
    def act(self, obs, params):
        turn_angle = self.rng.normal(0.0, params.turn_noise)
        accel = self.rng.normal(0.0, 5.0)
        return turn_angle, accel

class PathFollow(Behaviour):
    def __init__(self, path, seed):
        super().__init__(seed)
        self.path = path
        self.s = 0.0               # path parameter
        self.s_speed = 0.5         # how fast we move along the path
        self.epsilon = 0.05        # deviation probability
    
    def act(self, obs, params):
        if self.path is None:
            return self.random_policy(obs)

        pos = obs["pos"]
        direction = obs["direction"]

        center = self.path.center
        radius = self.path.radius

        # vector from center to agent
        rel = pos - center
        rel[2] = 0.0
        dist = np.linalg.norm(rel)

        if dist < 1e-6:
            return 0.0, 0.0

        # closest point on the circle
        radial_dir = rel / dist
        closest_point = center + radius * radial_dir

        # tangent direction (CCW)
        tangent = np.array([
            -radial_dir[1],
            radial_dir[0],
            0.0
        ])

        # path attraction
        to_path = closest_point - pos
        Kp = 1.0

        desired = tangent + Kp * to_path
        desired /= np.linalg.norm(desired)

        # heading error
        cross = direction[0] * desired[1] - direction[1] * desired[0]
        dot = np.dot(direction[:2], desired[:2])
        turn_angle = np.arctan2(cross, dot)

        # mild epsilon disturbance
        if self.rng.random() < params.epsilon:
            turn_angle += self.rng.normal(0.0, params.turn_noise * 0.3)

        desired_speed = 0.7 * params.max_speed
        accel = desired_speed - obs["speed"]

        return turn_angle, accel

class POI(Behaviour):
    def __init__(self, pois, seed):
        super().__init__(seed)
        self.pois = pois
        self.poi_idx = None
    
    def _select_poi_idx(self):
        # Random initial target
        if self.poi_idx is None:
            return int(self.rng.integers(0, len(self.pois)))

        # Pick a different index than current
        choices = list(range(len(self.pois)))
        choices.remove(self.poi_idx)
        return int(self.rng.choice(choices))

    def act(self, obs, params):
        if not self.pois:
            print("No points of interest!")
            raise ValueError

        # Select initial target
        if self.poi_idx is None:
            self.poi_idx = self._select_poi_idx()

        target = self.pois[self.poi_idx]
        to_target = target - obs["pos"]
        to_target[2] = 0.0

        dist = float(np.linalg.norm(to_target[:2]))

        # Arrived at point
        if dist < POI_REACHED_EPS and POI_SWITCH_ON_REACH and len(self.pois) > 1:
            self.poi_idx = self._select_poi_idx()
            target = self.pois[self.poi_idx]
            to_target = target - obs["pos"]
            to_target[2] = 0.0
            dist = float(np.linalg.norm(to_target[:2]))

        # Calculate desired angle and error
        desired_dir = to_target / (np.linalg.norm(to_target) + 1e-12)

        cur = obs["direction"]
        cur[2] = 0.0
        cur /= (np.linalg.norm(cur) + 1e-12)

        cross = cur[0] * desired_dir[1] - cur[1] * desired_dir[0]
        dot = cur[0] * desired_dir[0] + cur[1] * desired_dir[1]
        angle_error = float(np.arctan2(cross, dot))

        # Set control parameters
        noise = self.rng.normal(0.0, params.turn_noise) * NOISE_SCALE
        turn_angle = YAW_GAIN * angle_error + noise
        accel = ACCEL_GAIN * (params.max_speed - obs["speed"])

        return turn_angle, accel