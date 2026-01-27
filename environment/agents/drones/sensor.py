import numpy as np
from utils.vec_utils import *

class Sensor:
    def sense(self, drone, animals):
        raise NotImplementedError

class Camera(Sensor):
    def __init__(self, hfov_rad, vfov_rad, near=0.0, far=float("inf")):
        self.hfov = hfov_rad
        self.vfov = vfov_rad
        self.near = near
        self.far = far

        # FOV limits
        self.tan_h = np.tan(hfov_rad * 0.5)
        self.tan_v = np.tan(vfov_rad * 0.5)

    def sense(self, drone, points):
        inside, cam_xyz = self.points_in_view_frustum(
            points=points,
            cam_pos=drone.pos,
            view_dir=drone.view_dir,
        )

        return inside, cam_xyz

    def points_in_view_frustum(self, points, cam_pos, view_dir):
        # Normalize forward
        forward = unit(view_dir)

        # Right is constrained to XY plane
        right = np.cross(WORLD_UP, forward)
        right_norm = np.linalg.norm(right)

        if right_norm < 1e-8:
            # Looking straight up/down -> arbitrary right in XY plane
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= right_norm

        # Complete orthonormal basis
        up = np.cross(forward, right)

        # Vector from camera to points
        v = points - cam_pos  # (N, 3)

        # Project into camera space
        x = np.dot(v, right)
        y = np.dot(v, up)
        z = np.dot(v, forward)

        cam_xyz = np.stack([x, y, z], axis=1)

        # Reject points behind camera
        valid = z > 0.0

        inside = (
            valid &
            (z >= self.near) &
            (z <= self.far) &
            (np.abs(x) <= z * self.tan_h) &
            (np.abs(y) <= z * self.tan_v)
        )

        return inside, cam_xyz