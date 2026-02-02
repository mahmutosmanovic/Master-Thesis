from settings import *
import numpy as np


class Drone:

    def __init__(self, config, animal_pos):

        self.config = config
        self.z = 40

        # Fixed camera mounting direction
        v = np.array([-1, -1, -1], dtype=float)
        self.camera_dir = v / np.linalg.norm(v)

        # Body yaw (radians)
        self.yaw = 0.0

        self._init_pos(animal_pos)

    # ----------------------------------

    def _init_pos(self, animal_pos):

        ax, ay, az = animal_pos

        min_r = getattr(self.config, "spawn_min_radius", self.z * 0.5)
        max_r = getattr(self.config, "spawn_max_radius", self.z)

        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(min_r, max_r)

        self.x = ax + r * np.cos(theta)
        self.y = ay + r * np.sin(theta)

    # ----------------------------------

    def _get_view_dir(self):
        """
        Rotate camera_dir by yaw
        """

        c = np.cos(self.yaw)
        s = np.sin(self.yaw)

        Rz = np.array([
            [ c, -s, 0],
            [ s,  c, 0],
            [ 0,  0, 1]
        ])

        return Rz @ self.camera_dir

    # ----------------------------------

    def observe(self, animal):

        drone_pos = np.array([self.x, self.y, self.z])

        animal_pos = np.array(
            [animal.x, animal.y, animal.z],
            dtype=float
        )

        # View direction (with yaw)
        view_dir = self._get_view_dir()
        view_dir /= np.linalg.norm(view_dir)

        # Vector to animal
        to_animal = animal_pos - drone_pos
        dist = np.linalg.norm(to_animal)

        max_dist = 100.0

        # Default outputs
        in_view = 0
        angle_score = 0.0
        dist_score = 0.0

        # Edge case
        if dist == 0:
            return (1, 1.0, 1.0)

        # Too far
        if dist > max_dist:
            return (0, 0.0, 0.0)

        # Normalize
        to_animal /= dist

        # Angle
        cos_angle = np.dot(view_dir, to_animal)

        half_fov = self.config.fov / 2
        cos_threshold = np.cos(half_fov)

        # Distance score
        dist_score = 1.0 - dist / max_dist
        dist_score = np.clip(dist_score, 0.0, 1.0)

        # Outside FOV
        if cos_angle < cos_threshold:
            return (0, 0.0, 0.0)

        # Inside FOV
        in_view = 1

        # Angle score
        angle_score = (cos_angle - cos_threshold) / (1 - cos_threshold)
        angle_score = np.clip(angle_score, 0.0, 1.0)

        return (in_view, angle_score, dist_score)

    # ----------------------------------

    def policy(self, obs):

        print(obs)
        # Random yaw for now

        dx = (np.random.random() - 0.5) * 10
        dy = (np.random.random() - 0.5) * 10
        dz = (np.random.random() - 0.5) * 10
        dyaw = np.random.uniform(-1, 1)

        return dx, dy, dz, dyaw

    # ----------------------------------

    def step(self, action):

        dx, dy, dz, dyaw = action

        # Yaw control
        self.yaw += dyaw * self.config.yaw_speed
        self.yaw = (self.yaw + np.pi) % (2*np.pi) - np.pi

        # Movement
        self.x += dx
        self.y += dy
        self.z += dz
