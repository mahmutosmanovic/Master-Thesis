import matplotlib
matplotlib.use("Agg")

import time
import numpy as np
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def _normalize(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-8)


def _dummy_segment():
    return [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)]


def _frustum_segments(pos, forward, hfov, vfov, depth):
    pos = np.asarray(pos, dtype=float)
    forward = _normalize(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)

    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])

    right = _normalize(right)
    up = _normalize(np.cross(right, forward))

    h = np.tan(hfov * 0.5) * depth
    v = np.tan(vfov * 0.5) * depth

    center = pos + forward * depth

    corners = [
        center + right * h + up * v,
        center - right * h + up * v,
        center - right * h - up * v,
        center + right * h - up * v,
    ]

    segs = []

    for c in corners:
        segs.append(np.array([pos, c], dtype=float))

    for i in range(4):
        segs.append(np.array([corners[i], corners[(i + 1) % 4]], dtype=float))

    return segs


def _drone_cross(pos, forward, size, rotor_radius=None, n_circle_pts=24):
    pos = np.asarray(pos, dtype=float)
    forward = _normalize(forward)

    # keep marker in horizontal plane
    f = forward.copy()
    f[2] = 0.0

    if np.linalg.norm(f) < 1e-6:
        f = np.array([1.0, 0.0, 0.0])

    f = _normalize(f)

    right = np.cross(f, [0.0, 0.0, 1.0])
    right = _normalize(right)

    d1 = _normalize(f + right)
    d2 = _normalize(f - right)

    p1 = pos + d1 * size
    p2 = pos - d1 * size
    p3 = pos + d2 * size
    p4 = pos - d2 * size

    segs = [
        np.array([p1, p2], dtype=float),
        np.array([p3, p4], dtype=float),
    ]

    if rotor_radius is None:
        rotor_radius = 0.22 * size

    t = np.linspace(0.0, 2.0 * np.pi, n_circle_pts)
    circle_xy = np.stack(
        [
            rotor_radius * np.cos(t),
            rotor_radius * np.sin(t),
            np.zeros_like(t),
        ],
        axis=1
    )

    for center in [p1, p2, p3, p4]:
        segs.append(center + circle_xy)

    return segs


def _animal_geometry(pos, forward, size, n_body_pts=48, n_head_pts=36):
    """
    Animal glyph geometry:
    - filled ellipse body
    - outline-only circular head
    """
    pos = np.asarray(pos, dtype=float)
    forward = np.asarray(forward, dtype=float)

    f = forward.copy()
    f[2] = 0.0

    if np.linalg.norm(f) < 1e-6:
        f = np.array([1.0, 0.0, 0.0])

    f = _normalize(f)

    up = np.array([0.0, 0.0, 1.0])
    right = _normalize(np.cross(f, up))

    # body
    body_center = pos + up * (0.14 * size)
    body_len = 1.0 * size
    body_wid = 0.48 * size

    t = np.linspace(0.0, 2.0 * np.pi, n_body_pts, endpoint=False)
    body_pts = np.array([
        body_center
        + body_len * np.cos(tt) * f
        + body_wid * np.sin(tt) * right
        for tt in t
    ])

    # head
    head_center = (
        body_center
        + 1.15 * size * f
        + 0.18 * size * up
    )
    head_r = 0.42 * size

    th = np.linspace(0.0, 2.0 * np.pi, n_head_pts)
    head_pts = np.array([
        head_center
        + head_r * np.cos(tt) * right
        + head_r * np.sin(tt) * up
        for tt in th
    ])

    return body_pts, head_pts


class Viewer:
    def __init__(self, config):
        self.dt = config["dt"]

        self.interval_ms = int(self.dt * 1000)
        self.fps = int(1 / self.dt) * 4

        self.drone_cfg = config["drone"]
        self.config = config

        self.frames = []
        self.recording = False

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

        visible = np.zeros(len(animals), dtype=bool)

        if fov is not None and len(animals) > 0:
            animal_obs = fov[:, 4:].reshape(
                len(drones),
                len(animals),
                self.config.model.space.animal_features
            )
            visible = np.any(animal_obs[:, :, 0] == 1.0, axis=0)

        self.frames.append(dict(
            drones=np.array([d.pos.to_numpy() for d in drones], dtype=float),
            drone_types=[d.drone_type for d in drones],
            animals=np.array([a.pos.to_numpy() for a in animals], dtype=float),
            views=np.array([d.view_dir.to_numpy() for d in drones], dtype=float),
            visible=visible
        ))

    def stop_recording(self):
        if not self.frames:
            return

        print(f"[Viewer] Rendering video ({len(self.frames)} frames)...")

        frames = self.frames

        save_dir = Path(__file__).parent.parent / "recordings"
        save_dir.mkdir(exist_ok=True)

        filename = save_dir / f"recording_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection="3d")

        ax.set_proj_type("ortho")
        ax.view_init(30, -55)

        pts = np.concatenate(
            [f["drones"] for f in frames] + [f["animals"] for f in frames],
            axis=0
        )

        low, high = pts.min(0), pts.max(0)
        span = high - low
        span = np.maximum(span, 1e-3)
        pad = 0.05 * span

        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
        ax.set_zlim(low[2] - pad[2], high[2] + pad[2])

        ax.set_box_aspect(span + 2 * pad)

        sky = (0.84, 0.90, 0.96, 1.0)
        ground = (0.46, 0.66, 0.34, 1.0)

        fig.patch.set_facecolor(sky)
        ax.set_facecolor(sky)

        ax.xaxis.pane.set_facecolor(sky)
        ax.yaxis.pane.set_facecolor(sky)
        ax.zaxis.pane.set_facecolor(ground)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        vivid = ["#384047", "#ff3b30", "#ffd60a", "#af52de"]

        type_style = {
            t: dict(
                color=vivid[i % len(vivid)],
                hfov=np.deg2rad(cfg["hor_angle"]),
                vfov=np.deg2rad(cfg["ver_angle"]),
                depth=cfg["view_range"] / 10,
                f_color=vivid[i % len(vivid)],
            )
            for i, (t, cfg) in enumerate(self.drone_cfg.items())
        }

        n_drones = len(frames[0]["drones"])
        n_animals = len(frames[0]["animals"])
        drone_types = frames[0]["drone_types"]

        trail_len = 200
        d_hist = deque(maxlen=trail_len)
        a_hist = deque(maxlen=trail_len)

        drone_trails = [
            ax.plot(
                [], [], [],
                lw=0.7,
                color=type_style[drone_types[j]]["color"],
                alpha=0.7
            )[0]
            for j in range(n_drones)
        ]

        animal_trails = [
            ax.plot([], [], [], lw=1.2, alpha=0.55)[0]
            for _ in range(n_animals)
        ]

        cross_collection = Line3DCollection(
            _dummy_segment(),
            linewidths=2.0,
            colors="#222222",
            alpha=1.0
        )
        ax.add_collection3d(cross_collection)

        frustum_collection = Line3DCollection(
            _dummy_segment(),
            linewidths=1.0,
            alpha=0.6
        )
        ax.add_collection3d(frustum_collection)

        animal_head_collection = Line3DCollection(
            _dummy_segment(),
            linewidths=1.6,
            alpha=0.98
        )
        ax.add_collection3d(animal_head_collection)

        animal_body_collection = Poly3DCollection(
            [],
            alpha=0.90,
            linewidths=0.0
        )
        ax.add_collection3d(animal_body_collection)

        animal_color_visible = "#8c6a3b"
        animal_color_hidden = "#5a4632"

        span_xy = max(span[0], span[1])
        drone_size = 0.04 * span_xy
        animal_size = 0.032 * span_xy

        def update(i):
            f = frames[i]

            d = f["drones"]
            a = f["animals"]
            v = f["views"]
            visible = f["visible"]

            d_hist.append(d)
            a_hist.append(a)

            dh = np.array(d_hist)
            ah = np.array(a_hist)

            for j in range(n_drones):
                drone_trails[j].set_data(dh[:, j, 0], dh[:, j, 1])
                drone_trails[j].set_3d_properties(dh[:, j, 2])

            for j in range(n_animals):
                trail_color = (
                    animal_color_visible if visible[j] else animal_color_hidden
                )
                animal_trails[j].set_data(ah[:, j, 0], ah[:, j, 1])
                animal_trails[j].set_3d_properties(ah[:, j, 2])
                animal_trails[j].set_color(trail_color)

            # drone glyphs
            segs = []
            colors = []

            for j in range(n_drones):
                drone_segs = _drone_cross(
                    d[j],
                    v[j],
                    drone_size,
                    rotor_radius=0.22 * drone_size,
                    n_circle_pts=28
                )
                segs += drone_segs
                colors += ["#333333"] * len(drone_segs)

            cross_collection.set_segments(segs)
            cross_collection.set_color(colors)

            # drone frustums
            fsegs = []
            fcolors = []

            for j in range(n_drones):
                s = type_style[f["drone_types"][j]]
                fr = _frustum_segments(
                    d[j],
                    v[j],
                    s["hfov"],
                    s["vfov"],
                    s["depth"]
                )
                fsegs += fr
                fcolors += [s["f_color"]] * len(fr)

            frustum_collection.set_segments(fsegs)
            frustum_collection.set_color(fcolors)

            # animal glyphs
            body_polys = []
            body_colors = []
            head_segs = []
            head_colors = []

            for j in range(n_animals):
                if i > 0:
                    motion = a[j] - frames[i - 1]["animals"][j]
                elif len(frames) > 1:
                    motion = frames[1]["animals"][j] - a[j]
                else:
                    motion = np.array([1.0, 0.0, 0.0])

                if np.linalg.norm(motion[:2]) < 1e-8:
                    motion = np.array([1.0, 0.0, 0.0])

                body_pts, head_pts = _animal_geometry(a[j], motion, animal_size)

                color = (
                    animal_color_visible if visible[j] else animal_color_hidden
                )

                body_polys.append(body_pts)
                body_colors.append(color)

                head_segs.append(head_pts)
                head_colors.append(color)

            animal_body_collection.set_verts(body_polys)
            animal_body_collection.set_facecolor(body_colors)
            animal_body_collection.set_edgecolor("none")

            animal_head_collection.set_segments(head_segs)
            animal_head_collection.set_color(head_colors)

            return []

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=self.interval_ms
        )

        anim.save(
            filename,
            writer=FFMpegWriter(fps=self.fps)
        )

        plt.close(fig)

        print("[Viewer] saved:", filename.name)

        self.recording = False
        self.frames = []

    def close(self):
        self.stop_recording()