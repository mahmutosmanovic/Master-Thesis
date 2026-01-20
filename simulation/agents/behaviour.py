import numpy as np
from simulation.settings import *
from abc import ABC, abstractmethod

def elevation_angle(v):
    horiz = np.linalg.norm(v[:2])
    return np.arctan2(v[2], horiz)

def signed_yaw_error(cur, des):
    cur_xy = cur[:2]
    des_xy = des[:2]

    cur_xy /= np.linalg.norm(cur_xy) + 1e-12
    des_xy /= np.linalg.norm(des_xy) + 1e-12

    cross = cur_xy[0] * des_xy[1] - cur_xy[1] * des_xy[0]
    dot   = cur_xy[0] * des_xy[0] + cur_xy[1] * des_xy[1]
    return np.arctan2(cross, dot)

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
        yaw_rate = self.rng.normal(0.0, params.turn_noise)
        pitch_rate = self.rng.normal(0.0, params.turn_noise)
        accel = self.rng.normal(0.0, 5.0)
        return yaw_rate, pitch_rate, accel

class PathFollow(Behaviour):
    def __init__(self, path, seed):
        super().__init__(seed)
        self.path = path
        self.s = 0.0              # path parameter
        self.s_speed = 0.5        # how fast to move along path
        self.epsilon = 0.05

        self.Kp = 1.0             # attraction strength to path
    
    def act(self, obs, params):
        if self.path is None:
            raise NotImplementedError

        pos = obs["pos"]
        cur = obs["direction"]

        # path reference
        p = self.path.position(self.s)
        t = self.path.tangent(self.s)

        # attraction to path
        to_path = p - pos

        desired = t + self.Kp * to_path
        desired /= np.linalg.norm(desired) + 1e-12

        # yaw
        yaw_error = signed_yaw_error(cur, desired)
        yaw_rate = yaw_error

        # pitch
        cur_pitch = elevation_angle(cur)
        des_pitch = elevation_angle(desired)
        pitch_rate = des_pitch - cur_pitch

        # exploration noise
        if self.rng.random() < params.epsilon:
            yaw_rate += self.rng.normal(0.0, params.turn_noise * 0.3)
            pitch_rate += self.rng.normal(0.0, params.turn_noise * 0.3)

        # speed
        desired_speed = 0.7 * params.max_speed
        accel = desired_speed - obs["speed"]

        # advance path parameter (in alignment with tangent)
        t = self.path.tangent(self.s)
        v = obs["direction"] * obs["speed"]

        ds = max(0.0, np.dot(v, t))
        self.s += self.s_speed * ds

        return yaw_rate, pitch_rate, accel

class POI(Behaviour):
    def __init__(self, pois, seed):
        super().__init__(seed)
        self.pois = pois
        self.poi_idx = None
    
    def _select_poi_idx(self):
        if self.poi_idx is None:
            return int(self.rng.integers(0, len(self.pois)))
        choices = list(range(len(self.pois)))
        choices.remove(self.poi_idx)
        return int(self.rng.choice(choices))

    def act(self, obs, params):
        if not self.pois:
            raise ValueError("No points of interest!")

        if self.poi_idx is None:
            self.poi_idx = self._select_poi_idx()

        target = self.pois[self.poi_idx]
        to_target = target - obs["pos"]
        dist = np.linalg.norm(to_target)

        if dist < POI_REACHED_EPS and POI_SWITCH_ON_REACH and len(self.pois) > 1:
            self.poi_idx = self._select_poi_idx()
            target = self.pois[self.poi_idx]
            to_target = target - obs["pos"]
            dist = np.linalg.norm(to_target)

        desired = to_target / (dist + 1e-12)
        cur = obs["direction"]

        # yaw
        yaw_error = signed_yaw_error(cur, desired)

        # pitch
        cur_pitch = elevation_angle(cur)
        des_pitch = elevation_angle(desired)
        pitch_error = des_pitch - cur_pitch

        noise = self.rng.normal(0.0, params.turn_noise) * NOISE_SCALE
        yaw_rate = YAW_GAIN * yaw_error + noise
        pitch_rate = PITCH_GAIN * pitch_error + noise

        accel = ACCEL_GAIN * (params.max_speed - obs["speed"])

        return yaw_rate, pitch_rate, accel