import matplotlib
matplotlib.use("Agg")

import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# Frustum construction
def _frustum_segments(pos, view_dir, hfov, vfov, depth):

    n = np.linalg.norm(view_dir)
    if n < 1e-8:
        return []

    forward = view_dir / n
    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn

    up = np.cross(right, forward)
    up /= (np.linalg.norm(up) + 1e-8)

    tan_h = np.tan(hfov * 0.5)
    tan_v = np.tan(vfov * 0.5)

    center = pos + forward * depth

    corners = [
        center + right*(depth*tan_h) + up*(depth*tan_v),
        center - right*(depth*tan_h) + up*(depth*tan_v),
        center - right*(depth*tan_h) - up*(depth*tan_v),
        center + right*(depth*tan_h) - up*(depth*tan_v),
    ]

    segs = []

    # rays
    for c in corners:
        segs.append(((pos[0], c[0]), (pos[1], c[1]), (pos[2], c[2])))

    # far rectangle
    loop = corners + [corners[0]]
    for a, b in zip(loop[:-1], loop[1:]):
        segs.append(((a[0], b[0]), (a[1], b[1]), (a[2], b[2])))

    return segs


# Viewer
class Viewer:

    def __init__(self, config):

        # timing
        self.dt = config["dt"]
        self.interval_ms = int(self.dt * 1000)
        self.fps = int(1 / self.dt)

        # drone camera config
        drone_cfg = config["drone"]["init"]

        self.hfov = np.deg2rad(drone_cfg["hor_angle"])
        self.vfov = np.deg2rad(drone_cfg["ver_angle"])
        self.view_range = drone_cfg["view_range"]

        # shorter visual frustum (looks nicer)
        self.frustum_depth = 0.25 * self.view_range

        # recording
        self.frames = []
        self.recording = False

    # Record simulation frames
    def _start_recording(self):
        self.frames = []
        self.recording = True
        print("[Viewer] Recording frames...")

    def draw(self, drones, animals, mode):

        if mode is None:
            if self.recording:
                self.stop_recording()
            return

        if not self.recording:
            self._start_recording()

        self.frames.append({
            "drones": np.array([d.pos.to_numpy() for d in drones]),
            "animals": np.array([a.pos.to_numpy() for a in animals]),
            "views": np.array([d.view_dir.to_numpy() for d in drones]),
        })

    # Render video
    def stop_recording(self):

        if not self.frames:
            return

        print("[Viewer] Rendering video...")

        save_dir = Path(__file__).parent.parent / "recordings"
        save_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = save_dir / f"recording_{timestamp}.mp4"

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")

        # GLOBAL LIMITS
        pts = np.concatenate(
            [f["drones"] for f in self.frames] +
            [f["animals"] for f in self.frames],
            axis=0
        )

        low = pts.min(axis=0)
        high = pts.max(axis=0)

        def set_axes_equal():
            mid = (low + high) / 2
            r = np.max(high - low) / 2
            ax.set_xlim(mid[0]-r, mid[0]+r)
            ax.set_ylim(mid[1]-r, mid[1]+r)
            ax.set_zlim(mid[2]-r, mid[2]+r)

        drone_trail = []
        animal_trail = []

        def update(frame_idx):

            frame = self.frames[frame_idx]
            drones = frame["drones"]
            animals = frame["animals"]
            views = frame["views"]

            drone_trail.append(drones)
            animal_trail.append(animals)

            ax.cla()
            set_axes_equal()

            dh = np.array(drone_trail)
            ah = np.array(animal_trail)

            # trails
            for i in range(dh.shape[1]):
                ax.plot(dh[:, i, 0], dh[:, i, 1], dh[:, i, 2],
                        color="red", alpha=0.4)

            for i in range(ah.shape[1]):
                ax.plot(ah[:, i, 0], ah[:, i, 1], ah[:, i, 2],
                        color="green", alpha=0.4)

            # agents 
            ax.scatter(drones[:, 0], drones[:, 1], drones[:, 2],
                       c="red", s=80)

            ax.scatter(animals[:, 0], animals[:, 1], animals[:, 2],
                       c="green", s=50)

            # camera + frustum 
            for p, v in zip(drones, views):

                v = v / (np.linalg.norm(v) + 1e-8)

                # direction arrow
                ax.quiver(
                    p[0], p[1], p[2],
                    v[0], v[1], v[2],
                    length=25,
                    color="blue"
                )

                segs = _frustum_segments(
                    p, v,
                    self.hfov,
                    self.vfov,
                    self.frustum_depth
                )

                for xs, ys, zs in segs:
                    ax.plot(xs, ys, zs,
                            color="blue",
                            alpha=0.25,
                            linewidth=1.2)

            ax.set_title("Inference Playback")

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.frames),
            interval=self.interval_ms,
            blit=False
        )

        writer = FFMpegWriter(fps=self.fps)
        anim.save(filename, writer=writer)

        plt.close(fig)

        print(f"[Viewer] Saved: {filename.name}")
        self.recording = False

    def close(self):
        self.stop_recording()
