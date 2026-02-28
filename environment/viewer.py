import matplotlib
matplotlib.use("Agg")

import time
import numpy as np
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
        self.fps = int(1 / self.dt)*4

        self.drone_cfg = config["drone"]

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
            "drone_types": [d.drone_type for d in drones],
            "animals": np.array([a.pos.to_numpy() for a in animals]),
            "views": np.array([d.view_dir.to_numpy() for d in drones]),
            "visible": visible,
            "reward": reward,
        })

    def stop_recording(self):
        if not self.frames:
            return

        print(f"[Viewer] Rendering video ({len(self.frames)} frames)...")
        frames = self.frames
        
        # 1. Setup Filepath
        save_dir = Path(__file__).parent.parent / "recordings"
        save_dir.mkdir(exist_ok=True)
        filename = save_dir / f"recording_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"

        # 2. Figure Setup
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(projection="3d")
        ax.set_proj_type("ortho")
        
        # Calculate global bounds so the camera doesn't jump/zoom mid-video
        pts = np.concatenate([f["drones"] for f in frames] + [f["animals"] for f in frames], axis=0)
        low, high = pts.min(0), pts.max(0)
        pad = 0.1 * (high - low + 1e-6)
        ax.set_xlim(low[0]-pad[0], high[0]+pad[0])
        ax.set_ylim(low[1]-pad[1], high[1]+pad[1])
        ax.set_zlim(low[2]-pad[2], high[2]+pad[2])
        ax.set_box_aspect((high-low+2*pad))

        # 3. Styling Map (using your Enum and __init__ variables)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        type_style = {
            drone_type: {
                "color": colors[i % len(colors)], 
                "hfov": np.deg2rad(drone_type_cfg["hor_angle"]), 
                "vfov": np.deg2rad(drone_type_cfg["ver_angle"]),
                "depth": drone_type_cfg["view_range"]/20,
                "f_color": colors[i % len(colors)]
            } for i, (drone_type, drone_type_cfg) in enumerate(self.drone_cfg.items())
        }
        
        legend_elements = []

        # drone types
        for i, (d_type, style) in enumerate(type_style.items()):
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=style["color"],
                    markeredgecolor='black',
                    markersize=6,
                    label=f"drone_{d_type}"
                )
            )

        # animal visibility
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w',
                markerfacecolor='green', markeredgecolor='black',
                label='Visible'),

            Line2D([0], [0], marker='o', color='w',
                markerfacecolor='red', markeredgecolor='black',
                label='Hidden'),
        ])

        ax.legend(handles=legend_elements,
                loc='upper left',
                bbox_to_anchor=(0.0, 0.9))

        # 4. Initialize Plot Objects
        n_drones = len(frames[0]["drones"])
        n_animals = len(frames[0]["animals"])
        drone_types = frames[0]["drone_types"]

        drone_scatter = ax.scatter([], [], [], s=80, edgecolors='black', linewidth=0.3)
        animal_scatter = ax.scatter([], [], [], s=60, depthshade=False)
        
        hud_text = ax.text2D(0.0185, 0.91, "", transform=ax.transAxes, family="monospace", 
                            bbox=dict(facecolor="white", alpha=0.3))

        drone_trails = [ax.plot([], [], [], color=type_style[drone_types[j]]["color"], 
                                lw=1.5, alpha=1.0)[0] for j in range(n_drones)]
        
        animal_trails = [ax.plot([], [], [], color="gray", lw=1.5, alpha=1.0)[0] 
                        for _ in range(n_animals)]

        # Use a Collection for frustums - much faster than creating 100+ Line3D objects
        dummy_seg = [np.array([[0, 0, 0], [0, 0, 0]])]
        frustum_collection = Line3DCollection(dummy_seg, colors="cornflowerblue", linewidths=0.7, alpha=0.6)
        ax.add_collection3d(frustum_collection)

        d_hist = deque(maxlen=self.drone_trail_len)
        a_hist = deque(maxlen=self.animal_trail_len)

        # 5. Animation Update
        def update(idx):
            f = frames[idx]
            d_pos, a_pos, views = f["drones"], f["animals"], f["views"]
            d_hist.append(d_pos)
            a_hist.append(a_pos)
            
            # HUD
            if f["reward"] is not None:
                hud_text.set_text(f"Frame: {idx:03d} \nReward: {f['reward']:.2f}")

            # Update Scatters
            drone_scatter._offsets3d = (d_pos[:,0], d_pos[:,1], d_pos[:,2])
            drone_scatter.set_facecolors([type_style[t]["color"] for t in f["drone_types"]])
            
            animal_scatter._offsets3d = (a_pos[:,0], a_pos[:,1], a_pos[:,2])
            animal_scatter.set_facecolors(np.where(f["visible"], "green", "red"))

            # Update Trails
            dh, ah = np.array(d_hist), np.array(a_hist)
            for j in range(n_drones):
                drone_trails[j].set_data(dh[:, j, 0], dh[:, j, 1])
                drone_trails[j].set_3d_properties(dh[:, j, 2])
            for j in range(n_animals):
                animal_trails[j].set_data(ah[:, j, 0], ah[:, j, 1])
                animal_trails[j].set_3d_properties(ah[:, j, 2])

            # Update Frustums with dynamic FOV per drone type
            all_segs = []
            seg_colors = []
            for j in range(n_drones):
                style = type_style[f["drone_types"][j]]
                segs = _frustum_segments(d_pos[j], views[j], style["hfov"], 
                                        style["vfov"], style["depth"])
                all_segs.extend(segs)
                seg_colors.extend([style["f_color"]] * 8)

            frustum_collection.set_segments(all_segs)
            frustum_collection.set_colors(seg_colors)

            return [drone_scatter, animal_scatter, frustum_collection, hud_text] + drone_trails

        # 6. Save
        anim = FuncAnimation(fig, update, frames=len(frames), interval=self.interval_ms)
        anim.save(filename, writer=FFMpegWriter(fps=self.fps))
        plt.close(fig)
        
        print(f"[Viewer] Video saved: {filename.name}")
        self.recording = False

    def close(self):
        self.stop_recording()