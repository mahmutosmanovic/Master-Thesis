import numpy as np
from .agent import Agent
from dataclasses import dataclass
from environment.utils.vec_utils import unit, pitched_vector, WORLD_UP

@dataclass
class RewardMetrics:
    n_visible: int = 0
    min_distance: float = np.inf
    mean_distance: float = np.inf
    alignment_error: float = np.inf

class Drone(Agent):
    def __init__(self, agent_id, pos, direction, cfg, seed, yaw_rad=0.0, pos_scale=np.array([1.0, 1.0, 1.0])):
        super().__init__(agent_id, pos, direction, seed)
        self.cfg = cfg
        self.pos = pos
        self.pos_scale = pos_scale
        rad_camera_pitch = np.deg2rad(self.cfg.camera_pitch)
        self.view_dir = pitched_vector(rad_camera_pitch, yaw_rad)
        self.tan_h = np.tan(np.deg2rad(self.cfg.hfov) * 0.5)
        self.tan_v = np.tan(np.deg2rad(self.cfg.vfov) * 0.5)

    @property
    def action_dim(self):
        return 5
    
    @property
    def obs_dim(self) -> int:
        return 4 + self.camera_obs_dim

    def observe(self, animals) -> np.ndarray:
        self_obs = np.concatenate([
            # self.pos[[2]] / self.pos_scale[[2]],
            self.view_dir
        ]).astype(np.float32)

        return self_obs, self.camera_observe(self, animals)
    
    @property
    def camera_obs_dim(self) -> int:
        return self.cfg.max_targets * 4  # [u,v,z_norm,mask]

    def camera_observe(self, agent, targets) -> np.ndarray:
        points = np.array([animal.pos.copy() for animal in targets], dtype=float)

        inside, cam_xyz = self.points_in_view_frustum(
            points, agent.pos, agent.view_dir
        )

        vis = cam_xyz[inside]

        if vis.shape[0] > 0:
            d = np.linalg.norm(vis, axis=1)
            vis = vis[np.argsort(d)]

        feats = np.zeros((self.cfg.max_targets, 4), dtype=np.float32)
        m = min(self.cfg.max_targets, vis.shape[0])

        if m > 0:
            x, y, z = vis[:m, 0], vis[:m, 1], vis[:m, 2]
            u = x / (z * self.tan_h + 1e-8)
            v = y / (z * self.tan_v + 1e-8)

            z_norm = np.clip((z - self.cfg.near_plane) / (self.cfg.far_plane - self.cfg.near_plane + 1e-8), 0.0, 1.0)

            feats[:m, 0] = u
            feats[:m, 1] = v
            feats[:m, 2] = z_norm
            feats[:m, 3] = 1.0

        return feats.reshape(-1)

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
            (z >= self.cfg.near_plane) &
            (z <= self.cfg.far_plane) &
            (np.abs(x) <= z * self.tan_h) &
            (np.abs(y) <= z * self.tan_v)
        )

        return inside, cam_xyz
    
    def reward_metrics(self, obs: np.ndarray) -> RewardMetrics:
        feats = obs.reshape(self.cfg.max_targets, 4)

        mask = feats[:, 3] > 0.5 # targets in view
        if not np.any(mask):
            return RewardMetrics()

        u = feats[mask, 0]
        v = feats[mask, 1]
        z_norm = feats[mask, 2]

        d_norm = z_norm * np.sqrt(1.0 + (u*self.tan_h)**2 + (v*self.tan_v)**2)
        d_norm = np.clip(d_norm, 0.0, 1.0)

        center_error = float(np.sqrt(u*u + v*v).mean())

        return RewardMetrics(
            n_visible=np.sum(mask),
            min_distance=d_norm.min(),
            mean_distance=d_norm.mean(),
            alignment_error=center_error,
        )
    
    def update(self, action, dt):
        direction, norm_speed, view_yaw_rate = action
        self.apply_control(direction, norm_speed, dt)
        self.apply_view_yaw(view_yaw_rate, dt)
        self.move(dt)
    
    def apply_view_yaw(self, view_yaw_rate, dt):
        # Clamp yaw rate
        view_yaw_rate *= self.cfg.max_view_yaw
        view_yaw_rate = np.clip(
            view_yaw_rate,
            -self.cfg.max_view_yaw,
            self.cfg.max_view_yaw,
        )

        yaw = view_yaw_rate * dt
        if abs(yaw) < 1e-8:
            return

        c = np.cos(yaw)
        s = np.sin(yaw)

        # Rotation about Z axis
        Rz = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])

        self.view_dir = unit(Rz @ self.view_dir)
    
    def to_dict(self):
        view_x, view_y, view_z = self.view_dir
        return{
            **super().to_dict(),
            "view_x": view_x,
            "view_y": view_y,
            "view_z": view_z,
        }