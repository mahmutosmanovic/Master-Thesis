import numpy as np
from utils.vec_utils import *
from environment.settings import *

class Sensor:
    @property
    def obs_dim(self):
        raise NotImplementedError
    
    def get_obs(self, drone, animals):
        raise NotImplementedError
    
    def reward(self, drone, animals):
        raise NotImplementedError

class Camera(Sensor):
    def __init__(self, hfov_rad, vfov_rad, near=0.0, far=float("inf"), K=1, reward_scale=5.0, sigma=250, center_penalty_scale=1.0, center_sigma=0.35):
        self.hfov = hfov_rad
        self.vfov = vfov_rad
        self.near = near
        self.far = far
        self.K = K

        self.reward_scale = float(reward_scale)
        self.sigma = float(sigma)
        self.center_penalty_scale = float(center_penalty_scale)
        self.center_sigma = float(center_sigma)

        self.tan_h = np.tan(hfov_rad * 0.5)
        self.tan_v = np.tan(vfov_rad * 0.5)

    @property
    def obs_dim(self) -> int:
        return self.K * 4  # [u,v,z_norm,mask]

    def get_obs(self, drone, animals) -> np.ndarray:
        points = np.array([animal.pos.copy() for animal in animals], dtype=float)

        inside, cam_xyz = self.points_in_view_frustum(
            points, drone.pos, drone.view_dir
        )

        vis = cam_xyz[inside]

        if vis.shape[0] > 0:
            vis = vis[np.argsort(vis[:, 2])]  # sort by depth

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
            feats[:m, 3] = 1.0

        return feats.reshape(-1)

    def reward(self, drone, animals):
        if len(animals) == 0:
            return 0.0

        points = np.array([animal.pos.copy() for animal in animals], dtype=float)

        inside, cam_xyz = self.points_in_view_frustum(
            points, drone.pos, drone.view_dir
        )

        if not np.any(inside):
            return 0.0

        vis = cam_xyz[inside]
        d = np.linalg.norm(vis, axis=1)
        i = np.argmin(d)
        d = d[i]
        base = max(0.0, 1.0 - d / (self.sigma + 1e-8))
        distance_reward = float(self.reward_scale * base)
        
        x, y, z = float(vis[i, 0]), float(vis[i, 1]), float(vis[i, 2])
        u = x / (z * self.tan_h + 1e-8)
        v = y / (z * self.tan_v + 1e-8)

        center_dist = float(np.sqrt(u*u + v*v))
        penalty = self.center_penalty_scale * center_dist

        return distance_reward - penalty

    def points_in_view_frustum(self, points, cam_pos, view_dir):
        forward = unit(view_dir)

        right = np.cross(WORLD_UP, forward)
        n = np.linalg.norm(right)
        right = right / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])

        up = np.cross(right, forward)

        v = points - cam_pos

        x = np.dot(v, right)
        y = np.dot(v, up)
        z = np.dot(v, forward)

        cam_xyz = np.stack([x, y, z], axis=1)

        inside = (
            (z > 0.0) &
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

    def reward(self, drone, animals, sigma=250.0):
        if len(animals) == 0:
            raise ValueError("No observable animals")
        points = np.array([animal.pos.copy() for animal in animals], dtype=float)

        # nearest distance only
        dists = np.linalg.norm(points - drone.pos, axis=1)
        d = float(np.min(dists))

        # linear reward: closer is better
        base = max(0.0, 1.0 - d / (sigma + 1e-8))

        return self.reward_scale * base