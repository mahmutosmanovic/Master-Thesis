import numpy as np
from utils.vec_utils import *

class Sensor:
    def sense(self, drone, animals):
        raise NotImplementedError

class Camera(Sensor):
    def __init__(self, hfov_rad, vfov_rad, near=0.0, far=float("inf"), K=5):
        self.hfov = hfov_rad
        self.vfov = vfov_rad
        self.near = near
        self.far = far
        self.K = K

        self.tan_h = np.tan(hfov_rad * 0.5)
        self.tan_v = np.tan(vfov_rad * 0.5)

    @property
    def obs_dim(self) -> int:
        return self.K * 4  # [u,v,z_norm,mask] per slot

    def get_obs(self, drone, animals) -> np.ndarray:
        points = np.array([animal.pos.copy() for animal in animals], dtype=float)
        # assumes world.animals_pos is (N,3) array
        inside, cam_xyz = self.sense(drone, points)
        vis = cam_xyz[inside]

        if vis.shape[0] > 0:
            vis = vis[np.argsort(vis[:, 2])]  # sort by z

        feats = np.zeros((self.K, 4), dtype=np.float32)
        m = min(self.K, vis.shape[0])

        if m > 0:
            x, y, z = vis[:m, 0], vis[:m, 1], vis[:m, 2]
            u = x / (z * self.tan_h + 1e-8)
            v = y / (z * self.tan_v + 1e-8)
            z_norm = np.clip((z - self.near) / (self.far - self.near + 1e-8), 0.0, 1.0)

            feats[:m, 0] = u
            feats[:m, 1] = v
            feats[:m, 2] = z_norm
            feats[:m, 3] = 1.0  # mask

        return feats.reshape(-1)
    
    def reward(self, inside, cam_xyz, z_opt=10.0, sigma_center=0.6, sigma_z=6.0, empty_penalty=0.0):
        vis = cam_xyz[inside]  # (M,3)
        if vis.shape[0] == 0:
            return float(empty_penalty)

        x, y, z = vis[:, 0], vis[:, 1], vis[:, 2]

        u = x / (z * self.tan_h + 1e-8)
        v = y / (z * self.tan_v + 1e-8)

        center = np.exp(-(u*u + v*v) / (sigma_center**2 + 1e-8))
        rng    = np.exp(-((z - z_opt)**2) / (sigma_z**2 + 1e-8))
        score = center * rng
        return float(np.log1p(np.sum(score)))

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

class GPSSensor(Sensor):
    def __init__(
        self,
        max_targets: int,
        noise_pos: float = 0.0,
        pos_scale=1.0,          # scalar or (3,) array; used to normalize relative vectors
        reward_scale: float = 5.0,   # boosts signal vs disturbance penalty
        seed=None,
    ):
        self.max_targets = max_targets
        self.noise_pos = noise_pos
        self.pos_scale = np.asarray(pos_scale, dtype=np.float32)
        self.reward_scale = float(reward_scale)
        self.rng = np.random.default_rng(seed)

    @property
    def obs_dim(self) -> int:
        return self.max_targets * 7  # [dx,dy,dz,vx,vy,vz,mask] per slot

    def get_obs(self, drone, animals) -> np.ndarray:
        obs = np.zeros((self.max_targets, 7), dtype=np.float32)
        points = np.array([animal.pos.copy() for animal in animals], dtype=float)
        dirs = np.array([animal.direction for animal in animals], dtype=float)

        if len(points) == 0:
            raise ValueError("No observable animals")

        # distances to drone
        dists = np.linalg.norm(points - drone.pos, axis=1)
        k = min(len(points), self.max_targets)
        idx = np.argsort(dists)[:k]
        nearest = points[idx].astype(np.float32, copy=True)

        # optional noise on measured position
        if self.noise_pos > 0:
            nearest += self.rng.normal(0, self.noise_pos, size=nearest.shape).astype(np.float32)

        # relative vectors (target - drone)
        rel = nearest - drone.pos.astype(np.float32)
        dirs = dirs[idx]

        # normalize (supports scalar or (3,) vector)
        rel = rel / (self.pos_scale + 1e-8)
        # vels = vels[nearest] should be normalized

        obs[:k, 0:3] = rel
        obs[:k, 3:6] = dirs
        obs[:k, 6] = 1.0  # mask

        return obs.reshape(-1)

    def sense(self, drone, points):
        return True, points

    def reward(self, drone, animals, sigma=250.0):
        if len(animals) == 0:
            raise ValueError("No observable animals")
        points = np.array([animal.pos.copy() for animal in animals], dtype=float)

        # nearest distance only
        dists = np.linalg.norm(points - drone.pos, axis=1)
        d = float(np.min(dists))

        # linear reward: closer is better
        base = max(0.0, 1.0 - d / (sigma + 1e-8))

        return float(self.reward_scale * base)