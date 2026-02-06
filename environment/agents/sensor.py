import numpy as np
from utils.vec_utils import *
from dataclasses import dataclass

@dataclass
class SensorMetrics:
    n_visible: int = 0
    min_distance: float = np.inf
    mean_distance: float = np.inf
    alignment_error: float = np.inf

class Sensor:
    def __init__(self, sensor_scale=1, seed=None):
        self.sensor_scale = sensor_scale
        self.rng = np.random.default_rng(seed)

    @property
    def obs_dim(self):
        raise NotImplementedError
    
    def observe(self, drone, animals):
        raise NotImplementedError
    
    def metrics(self, obs):
        raise NotImplementedError

class Camera(Sensor):
    def __init__(
            self,
            hfov,
            vfov,
            near=0.0,
            far=np.inf,
            sensor_scale=1,
            max_targets=1,
            seed=None
        ):
        super().__init__(sensor_scale, seed)
        self.near = near
        self.far = far
        self.max_targets = max_targets

        self.tan_h = np.tan(np.deg2rad(hfov) * 0.5)
        self.tan_v = np.tan(np.deg2rad(vfov) * 0.5)

    @property
    def obs_dim(self) -> int:
        return self.max_targets * 4  # [u,v,z_norm,mask]

    def observe(self, agent, targets) -> np.ndarray:
        points = np.array([animal.pos.copy() for animal in targets], dtype=float)

        inside, cam_xyz = self.points_in_view_frustum(
            points, agent.pos, agent.view_dir
        )

        vis = cam_xyz[inside]

        if vis.shape[0] > 0:
            vis = vis[np.argsort(vis[:, 2])]  # sort by depth

        feats = np.zeros((self.max_targets, 4), dtype=np.float32)
        m = min(self.max_targets, vis.shape[0])

        if m > 0:
            x, y, z = vis[:m, 0], vis[:m, 1], vis[:m, 2]
            u = x / (z * self.tan_h + 1e-8)
            v = y / (z * self.tan_v + 1e-8)

            if self.far == np.inf:
                z_norm = np.clip(z / self.sensor_scale, 0.0, 1.0)
            else:
                z_norm = np.clip((z - self.near) / (self.far - self.near + 1e-8), 0.0, 1.0)

            feats[:m, 0] = u
            feats[:m, 1] = v
            feats[:m, 2] = z_norm
            feats[:m, 3] = 1.0

        return feats.reshape(-1)

    def metrics(self, obs: np.ndarray) -> SensorMetrics:
        feats = obs.reshape(self.max_targets, 4)

        mask = feats[:, 3] > 0.5 # targets in view
        if not np.any(mask):
            return SensorMetrics()

        u = feats[mask, 0]
        v = feats[mask, 1]
        z_norm = feats[mask, 2]

        min_dist = float(z_norm.min())
        mean_dist = float(z_norm.mean())

        center_error = float(np.sqrt(u*u + v*v).mean())

        return SensorMetrics(
            n_visible=np.sum(mask),
            min_distance=min_dist,
            mean_distance=mean_dist,
            alignment_error=center_error,
        )

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
        sensor_scale=1,
        seed=None,
    ):
        super().__init__(sensor_scale, seed)

        self.max_targets = max_targets
        self.noise_pos = noise_pos


    @property
    def obs_dim(self) -> int:
        return self.max_targets * 7  # [dx,dy,dz,vx,vy,vz,mask] per slot

    def observe(self, agent, targets) -> np.ndarray:
        obs = np.zeros((self.max_targets, 7), dtype=np.float32)
        points = np.array([animal.pos.copy() for animal in targets], dtype=float)
        dirs = np.array([animal.direction for animal in targets], dtype=float)

        if len(points) == 0:
            raise ValueError("No observable tragets")

        # distances to drone
        dists = np.linalg.norm(points - agent.pos, axis=1)
        k = min(len(points), self.max_targets)
        idx = np.argsort(dists)[:k]
        nearest = points[idx].astype(np.float32, copy=True)

        # optional noise on measured position
        if self.noise_pos > 0:
            nearest += self.rng.normal(0, self.noise_pos, size=nearest.shape).astype(np.float32)

        # relative vectors (target - drone)
        rel = nearest - agent.pos.astype(np.float32)
        dirs = dirs[idx]

        rel = rel / self.sensor_scale


        obs[:k, 0:3] = rel
        obs[:k, 3:6] = dirs
        obs[:k, 6] = 1.0  # mask

        return obs.reshape(-1)

    def metrics(self, obs: np.ndarray) -> SensorMetrics:
        feats = obs.reshape(self.max_targets, 7)

        mask = feats[:, 6] > 0.5
        n_visible = int(np.sum(mask))

        if n_visible == 0:
            return SensorMetrics()

        rel = feats[mask, 0:3]
        dists = np.linalg.norm(rel, axis=1)

        min_dist = float(dists.min())
        mean_dist = float(dists.mean())

        alignment_error = 0.0  # GPS has no bearing info by default

        return SensorMetrics(
            n_visible=n_visible,
            min_distance=min_dist,
            mean_distance=mean_dist,
            alignment_error=alignment_error,
        )