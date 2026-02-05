from settings import *

class Drone:
    """
    Drone kinematics (simple, PPO-friendly):
      - action = (dx, dy, dz, dyaw) in BODY frame
      - dx,dy,dz are per-step position deltas (already scaled by PPO action_scale)
      - dyaw is a per-step yaw command (already scaled by PPO action_scale), then multiplied by config.yaw_speed
    Observation (5D, learnable even when target is out of FoV):
      [rel_body_x/max_range, rel_body_y/max_range, rel_body_z/max_range, dist/max_range, fov_margin]
    """

    def __init__(self, config, animal_pos):
        self.config = config

        # Position
        self.x = 0.0
        self.y = 0.0
        self.z = 40.0

        # Body yaw (radians)
        self.yaw = 0.0

        # Fixed camera mounting direction in BODY frame
        v = np.array([1.0, 0.0, -0.5], dtype=np.float32)
        self.camera_dir = v / (np.linalg.norm(v) + 1e-8)

        self._init_pos(animal_pos)

    # ----------------------------

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _init_pos(self, animal_pos):
        ax, ay, az = animal_pos

        r_min = float(getattr(self.config, "spawn_min_radius", 40.0))
        r_max = float(getattr(self.config, "spawn_max_radius", 60.0))

        theta = np.random.uniform(0.0, 2.0 * np.pi)
        r = np.random.uniform(r_min, r_max)

        self.x = float(ax + r * np.cos(theta))
        self.y = float(ay + r * np.sin(theta))
        self.z = 40.0

        # yaw so animal roughly in front
        dx = float(ax - self.x)
        dy = float(ay - self.y)
        base_yaw = np.arctan2(dy, dx)

        yaw_noise = np.random.uniform(-0.15, 0.15)
        self.yaw = self._wrap_angle(base_yaw + yaw_noise)

    def reset(self, animal_pos):
        self._init_pos(animal_pos)

    # ----------------------------

    def _get_view_dir(self):
        """
        Camera view direction in WORLD frame.
        Rotate the BODY-frame camera_dir by yaw around z.
        """
        c = float(np.cos(self.yaw))
        s = float(np.sin(self.yaw))

        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        v = Rz @ self.camera_dir
        return v / (np.linalg.norm(v) + 1e-8)

    # ----------------------------

    def observe(self, animal):
        """
        Returns 5D observation:
          rel_body_norm (3), dist_norm (1), fov_margin (1)
        Works smoothly even when target is outside FoV.
        """
        drone_pos = np.array([self.x, self.y, self.z], dtype=np.float32)
        animal_pos = np.array([animal.x, animal.y, animal.z], dtype=np.float32)

        rel_world = animal_pos - drone_pos
        dist = float(np.linalg.norm(rel_world) + 1e-8)

        max_dist = float(self.config.max_view_range)

        # world -> body (undo yaw): body x=forward, y=left, z=up
        c = float(np.cos(self.yaw))
        s = float(np.sin(self.yaw))
        R_world_to_body = np.array([[ c,  s, 0.0],
                                   [-s,  c, 0.0],
                                   [0.0, 0.0, 1.0]], dtype=np.float32)

        rel_body = R_world_to_body @ rel_world
        rel_body_norm = rel_body / (max_dist + 1e-8)

        dist_norm = float(np.clip(dist / (max_dist + 1e-8), 0.0, 2.0))

        # FoV margin (continuous): cos(angle) - cos(threshold)
        view_dir = self._get_view_dir().astype(np.float32)
        to_animal = (rel_world / dist).astype(np.float32)

        cos_angle = float(np.clip(np.dot(view_dir, to_animal), -1.0, 1.0))

        fov = float(self.config.fov)
        # Safety: if user accidentally stored degrees, convert
        if fov > 3.2:  # ~> pi
            fov = np.deg2rad(fov)

        half_fov = 0.5 * fov
        cos_thr = float(np.cos(half_fov))

        fov_margin = cos_angle - cos_thr  # >0 inside FoV, <0 outside

        return np.array(
            [rel_body_norm[0], rel_body_norm[1], rel_body_norm[2], dist_norm, fov_margin],
            dtype=np.float32
        )

    # ----------------------------

    def step(self, action):
        """
        action = (dx, dy, dz, dyaw)
          - dx,dy,dz: BODY-frame deltas (already bounded by PPO tanh+scale)
          - dyaw: bounded by PPO tanh+scale, then multiplied by yaw_speed
        """
        dx, dy, dz, dyaw = action

        # No mismatch with PPO: clip only to the same bound PPO uses
        dyaw = float(np.clip(dyaw, -MAX_DYAW, MAX_DYAW))
        self.yaw = self._wrap_angle(self.yaw + dyaw * float(self.config.yaw_speed))

        c = float(np.cos(self.yaw))
        s = float(np.sin(self.yaw))

        # BODY -> WORLD
        vx = c * float(dx) - s * float(dy)
        vy = s * float(dx) + c * float(dy)

        self.x += vx
        self.y += vy
        self.z += float(dz)
