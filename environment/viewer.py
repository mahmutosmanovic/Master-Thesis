import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Polygon, Ellipse, Circle
import numpy as np


def _normalize(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-8)


def _heading_angle_deg(forward):
    f = np.asarray(forward, dtype=float).copy()
    f[2] = 0.0
    if np.linalg.norm(f) < 1e-8:
        f = np.array([1.0, 0.0, 0.0], dtype=float)
    f = _normalize(f)
    return np.degrees(np.arctan2(f[1], f[0]))


def _frustum_ground_polygon(pos, forward, hfov, depth):
    pos = np.asarray(pos, dtype=float)
    f = np.asarray(forward, dtype=float).copy()
    f[2] = 0.0

    if np.linalg.norm(f) < 1e-8:
        f = np.array([1.0, 0.0, 0.0], dtype=float)

    f = _normalize(f)
    right = _normalize(np.array([f[1], -f[0], 0.0], dtype=float))

    half_w = np.tan(hfov * 0.5) * depth
    far_center = pos + f * depth

    p0 = pos[:2]
    p1 = (far_center + right * half_w)[:2]
    p2 = (far_center - right * half_w)[:2]
    return np.vstack([p0, p1, p2])


def _drone_arm_endpoints_2d(pos, forward, size):
    pos = np.asarray(pos, dtype=float)
    f = np.asarray(forward, dtype=float).copy()
    f[2] = 0.0

    if np.linalg.norm(f) < 1e-8:
        f = np.array([1.0, 0.0, 0.0], dtype=float)

    f = _normalize(f)
    right = _normalize(np.array([f[1], -f[0], 0.0], dtype=float))

    d1 = _normalize(f + right)
    d2 = _normalize(f - right)

    p1 = pos + d1 * size
    p2 = pos - d1 * size
    p3 = pos + d2 * size
    p4 = pos - d2 * size

    arm1 = np.array([p1[:2], p2[:2]], dtype=float)
    arm2 = np.array([p3[:2], p4[:2]], dtype=float)
    rotor_centers = [p1[:2], p2[:2], p3[:2], p4[:2]]

    return arm1, arm2, rotor_centers


class Viewer:
    def __init__(self, config):
        self.config = config
        self.dt = float(config["dt"])

        # Only this controls saved-video speed
        self.video_speedup = 4.0
        self.fps = max(1, int(round(self.video_speedup / self.dt)))

        self.frames = []

        self.fig = None
        self.ax = None

        self.drone_trail_color = "tab:cyan"
        self.animal_trail_color = "tab:pink"

        self.animal_size = 12.5
        self.drone_size = 16.0

        # Smaller means more zoomed in
        self.view_padding_fraction = 0.035

        self.bg_outer = "#efefef"
        self.bg_inner = "#f7f7f7"
        self.border_color = "#d0d0d0"

    def _extract_visible_mask(self, drones, animals, fov):
        visible = np.zeros(len(animals), dtype=bool)


        if fov is None or len(animals) == 0:
            return visible
        try:
            animal_obs = fov[:, 4:4 + len(animals) * self.config.model.space.animal_features].reshape(
                len(drones),
                len(animals),
                self.config.model.space.animal_features,
            )
            visible = np.any(animal_obs[:, :, 0] == 1.0, axis=0)
        except Exception:
            pass

        return visible

    def _collect_frame(self, drones, animals, fov=None):
        self.frames.append(
            {
                "drones": np.array([d.pos.to_numpy() for d in drones], dtype=float),
                "drone_types": [d.drone_type for d in drones],
                "animals": np.array([a.pos.to_numpy() for a in animals], dtype=float),
                "views": np.array([d.view_dir.to_numpy() for d in drones], dtype=float),
                "visible": self._extract_visible_mask(drones, animals, fov),
            }
        )

    def _arena_bounds(self):
        if not self.frames:
            half = 250.0
            return -half, half, -half, half

        pts = []
        for frame in self.frames:
            if len(frame["drones"]) > 0:
                pts.append(frame["drones"][:, :2])
            if len(frame["animals"]) > 0:
                pts.append(frame["animals"][:, :2])

        if not pts:
            half = 250.0
            return -half, half, -half, half

        pts = np.vstack(pts)

        xmin = float(np.min(pts[:, 0]))
        xmax = float(np.max(pts[:, 0]))
        ymin = float(np.min(pts[:, 1]))
        ymax = float(np.max(pts[:, 1]))

        xspan = max(xmax - xmin, 1.0)
        yspan = max(ymax - ymin, 1.0)

        pad = self.view_padding_fraction
        xmin -= pad * xspan
        xmax += pad * xspan
        ymin -= pad * yspan
        ymax += pad * yspan

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * max(xmax - xmin, ymax - ymin)

        return cx - half, cx + half, cy - half, cy + half

    def _setup_axes(self, fig, ax):
        xmin, xmax, ymin, ymax = self._arena_bounds()

        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")

        fig.patch.set_facecolor(self.bg_outer)
        ax.set_facecolor(self.bg_outer)

        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor=self.bg_inner,
                edgecolor="none",
                zorder=0,
            )
        )

        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="none",
                edgecolor=self.border_color,
                linewidth=1.6,
                zorder=1,
            )
        )

    def _get_motion(self, frame_idx, obj_idx, kind="animals"):
        arr = self.frames[frame_idx][kind]

        if frame_idx > 0:
            motion = arr[obj_idx] - self.frames[frame_idx - 1][kind][obj_idx]
        elif len(self.frames) > 1:
            motion = self.frames[1][kind][obj_idx] - arr[obj_idx]
        else:
            motion = np.array([1.0, 0.0, 0.0], dtype=float)

        if np.linalg.norm(motion[:2]) < 1e-8:
            motion = np.array([1.0, 0.0, 0.0], dtype=float)

        return motion

    def _draw_soft_shadow(self, ax, x, y, angle_deg, width, height, alpha, dx, dy, zorder):
        ax.add_patch(
            Ellipse(
                (x + dx, y + dy),
                width=width,
                height=height,
                angle=angle_deg,
                facecolor="black",
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
            )
        )

    def _draw_animal(self, ax, pos, forward, visible):
        x, y = pos[:2]
        angle = _heading_angle_deg(forward)

        body_w = 2.2 * self.animal_size
        body_h = 1.15 * self.animal_size

        self._draw_soft_shadow(
            ax,
            x, y,
            angle,
            1.08 * body_w,
            0.95 * body_h,
            alpha=0.12,
            dx=4.5,
            dy=-3.5,
            zorder=3,
        )

        ax.add_patch(
            Ellipse(
                (x, y),
                width=body_w,
                height=body_h,
                angle=angle,
                facecolor="#57514a" if visible else "#6a645d",
                edgecolor="#1d1d1d",
                linewidth=1.1,
                alpha=0.98,
                zorder=6,
            )
        )

    def _draw_drone(self, ax, pos, forward, drone_type):
        x, y = pos[:2]

        f = np.asarray(forward, dtype=float).copy()
        f[2] = 0.0
        if np.linalg.norm(f) < 1e-8:
            f = np.array([1.0, 0.0, 0.0], dtype=float)
        f = _normalize(f)

        angle = _heading_angle_deg(f)

        self._draw_soft_shadow(
            ax,
            x, y,
            angle,
            1.9 * self.drone_size,
            1.2 * self.drone_size,
            alpha=0.10,
            dx=6.0,
            dy=-6.0,
            zorder=4,
        )

        cfg = self.config["drone"][drone_type]
        depth = float(cfg["view_range"]) / 6.0
        hfov = np.deg2rad(float(cfg["hor_angle"]))
        poly = _frustum_ground_polygon(pos, f, hfov, depth)

        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="black",
                edgecolor="none",
                alpha=0.035,
                zorder=4.5,
            )
        )

        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="none",
                edgecolor="black",
                linewidth=0.9,
                alpha=0.16,
                zorder=5,
            )
        )

        arm1, arm2, rotor_centers = _drone_arm_endpoints_2d(pos, f, self.drone_size)

        ax.plot(arm1[:, 0], arm1[:, 1], color="#222222", lw=1.6, alpha=0.96, zorder=8)
        ax.plot(arm2[:, 0], arm2[:, 1], color="#222222", lw=1.6, alpha=0.96, zorder=8)

        rotor_r = 0.22 * self.drone_size
        hole_r = 0.42 * rotor_r

        for cx, cy in rotor_centers:
            # outer ring
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius=rotor_r,
                    facecolor="none",
                    edgecolor="#222222",
                    linewidth=1.0,
                    alpha=0.96,
                    zorder=8,
                )
            )
            # inner hole
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius=hole_r,
                    facecolor=self.bg_inner,
                    edgecolor="#222222",
                    linewidth=0.8,
                    alpha=1.0,
                    zorder=8.1,
                )
            )

    def _draw_scene(self, fig, ax, frame_idx):
        self._setup_axes(fig, ax)

        frame = self.frames[frame_idx]
        drones = frame["drones"]
        animals = frame["animals"]
        views = frame["views"]
        visible = frame["visible"]

        for j in range(len(drones)):
            pts = np.array([f["drones"][j][:2] for f in self.frames[:frame_idx + 1]])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                lw=0.95,
                alpha=0.45,
                color=self.drone_trail_color,
                zorder=3,
            )

        for j in range(len(animals)):
            pts = np.array([f["animals"][j][:2] for f in self.frames[:frame_idx + 1]])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                lw=0.90,
                alpha=0.40,
                color=self.animal_trail_color,
                zorder=2.8,
            )

        for j in range(len(animals)):
            motion = self._get_motion(frame_idx, j, kind="animals")
            self._draw_animal(ax, animals[j], motion, bool(visible[j]))

        for j in range(len(drones)):
            self._draw_drone(ax, drones[j], views[j], frame["drone_types"][j])

    def draw(self, drones, animals, mode, fov=None, reward=None):
        if mode is None:
            return

        self._collect_frame(drones, animals, fov=fov)

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7.2, 7.2), dpi=140)

        self._draw_scene(self.fig, self.ax, len(self.frames) - 1)

        if self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

    def save_video(self, path):
        if not self.frames:
            return

        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)

        def update(i):
            self._draw_scene(fig, ax, i)
            return []

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.frames),
            interval=1,
            blit=False,
        )

        anim.save(path, writer=FFMpegWriter(fps=self.fps))
        plt.close(fig)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None