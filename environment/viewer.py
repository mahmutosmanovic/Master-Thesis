import matplotlib
matplotlib.use("Agg")

import time
import numpy as np
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

def _frustum_segments(pos, forward, hfov, vfov, depth):

    forward = forward / (np.linalg.norm(forward) + 1e-8)

    world_up = np.array([0, 0, 1.0])
    right = np.cross(forward, world_up)

    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])

    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    h = np.tan(hfov * 0.5) * depth
    v = np.tan(vfov * 0.5) * depth

    center = pos + forward * depth

    corners = [
        center + right*h + up*v,
        center - right*h + up*v,
        center - right*h - up*v,
        center + right*h - up*v,
    ]

    segs = []
    for c in corners:
        segs.append((pos, c))
    for i in range(4):
        segs.append((corners[i], corners[(i+1) % 4]))

    return segs

class Viewer:

    def __init__(self, config):

        self.dt = config["dt"]
        self.interval_ms = int(self.dt * 1000)
        self.fps = int(1 / self.dt)

        drone_cfg = config["drone"]["init"]

        self.hfov = np.deg2rad(drone_cfg["hor_angle"])
        self.vfov = np.deg2rad(drone_cfg["ver_angle"])
        self.frustum_depth = 10.0

        self.drone_trail_len = 200
        self.animal_trail_len = 200

        self.frames = []
        self.recording = False
        self._last_reward = None

    def _start_recording(self):
        self.frames = []
        self.recording = True
        print("[Viewer] Recording frames...")

    def draw(self, drones, animals, mode, fov=None, reward=None):

        if mode is None:
            if self.recording:
                self.stop_recording()
            return

        if not self.recording:
            self._start_recording()

        if fov is not None and len(animals) > 0:
            visible = np.any(fov[..., 0] == 1.0, axis=1)
        else:
            visible = np.zeros(len(animals), dtype=bool)

        # store frame
        self.frames.append({
            "drones": np.array([d.pos.to_numpy() for d in drones]),
            "animals": np.array([a.pos.to_numpy() for a in animals]),
            "views": np.array([d.view_dir.to_numpy() for d in drones]),
            "visible": visible,
            "reward": reward,
        })

    def stop_recording(self):

        if not self.frames:
            return

        print("[Viewer] Rendering video...")

        frames = self.frames

        save_dir = Path(__file__).parent.parent / "recordings"
        save_dir.mkdir(exist_ok=True)
        filename = save_dir / f"recording_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"

        # ---------- figure ----------
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")
        ax.grid(True)

        pts = np.concatenate(
            [f["drones"] for f in frames] +
            [f["animals"] for f in frames], axis=0)

        low, high = pts.min(0), pts.max(0)
        span = np.maximum(high - low, 1e-6)
        pad = 0.05 * span

        xmin, ymin, zmin = low - pad
        xmax, ymax, zmax = high + pad

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_box_aspect([xmax-xmin, ymax-ymin, zmax-zmin])

        # ---------- legend ----------
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='blue', label='Drone'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='green', label='Animal (Visible)'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', label='Animal (Not Visible)'),
            Line2D([0], [0], color='blue', lw=2, label='Camera FoV')
        ]

        # legend
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(0.304, 0.90),
        )

        drone_scatter = ax.scatter([], [], [], s=64, color="blue", depthshade=False)
        animal_scatter = ax.scatter([], [], [], s=64, depthshade=False)

        # reward
        hud_text = ax.text2D(
            0.0, 0.92, "",
            transform=ax.transAxes,
            fontsize=13,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.2, boxstyle="round,pad=0.4")
        )

        drone_hist = deque(maxlen=self.drone_trail_len)
        animal_hist = deque(maxlen=self.animal_trail_len)

        n_drones = frames[0]["drones"].shape[0]
        n_animals = frames[0]["animals"].shape[0]

        drone_trails = [ax.plot([], [], [], color="blue", lw=1)[0]
                        for _ in range(n_drones)]

        animal_trails = [ax.plot([], [], [], color="green", lw=1)[0]
                         for _ in range(n_animals)]

        frustum_lines = []

        def update(i):

            frame = frames[i]

            drones = frame["drones"]
            animals = frame["animals"]
            views = frame["views"]

            if frame["reward"] is not None:
                self._last_reward = frame["reward"]
            
            if self._last_reward is not None:
                hud_text.set_text(f"Reward: {self._last_reward:.2f}")

            drone_hist.append(drones)
            animal_hist.append(animals)

            dh = np.array(drone_hist)
            ah = np.array(animal_hist)

            drone_scatter._offsets3d = (drones[:,0], drones[:,1], drones[:,2])
            animal_scatter._offsets3d = (animals[:,0], animals[:,1], animals[:,2])

            cols = np.where(frame["visible"], "green", "red")
            cols_rgba = plt.cm.colors.to_rgba_array(cols)

            animal_scatter.set_facecolors(cols_rgba)
            animal_scatter.set_edgecolors(cols_rgba)
            animal_scatter._facecolor3d = cols_rgba
            animal_scatter._edgecolor3d = cols_rgba

            # trails
            for j, line in enumerate(drone_trails):
                line.set_data(dh[:, j, 0], dh[:, j, 1])
                line.set_3d_properties(dh[:, j, 2])

            for j, line in enumerate(animal_trails):
                line.set_data(ah[:, j, 0], ah[:, j, 1])
                line.set_3d_properties(ah[:, j, 2])

            # frustums
            for l in frustum_lines:
                l.remove()
            frustum_lines.clear()

            for p, v in zip(drones, views):
                for a, b in _frustum_segments(
                        p, v, self.hfov, self.vfov, self.frustum_depth):

                    line, = ax.plot(
                        [a[0], b[0]],
                        [a[1], b[1]],
                        [a[2], b[2]],
                        color="cornflowerblue",
                        lw=0.6,
                        alpha=0.98
                    )
                    frustum_lines.append(line)

            ax.set_title("Inference Playback")

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=self.interval_ms
        )

        anim.save(filename, writer=FFMpegWriter(fps=self.fps))
        plt.close(fig)

        print(f"[Viewer] Saved: {filename.name}")
        self.recording = False

    def close(self):
        self.stop_recording()