import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon, Ellipse, Circle
import numpy as np
import torch
from box import Box
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from environment import Env
from model import PPOAgent, MAPPOAgent, SACAgent, DQNAgent
from .run_utils import load_run


def _normalize(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-8)


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return float(default)
            return float(arr.mean())
        return float(x)
    except Exception:
        return float(default)


def _extract_metric(info, candidates, default=0.0):
    for key in candidates:
        if key in info:
            return _safe_float(info[key], default)
    return float(default)


def _frustum_ground_polygon(pos, forward, hfov, depth):
    pos = np.asarray(pos, dtype=float)
    forward = np.asarray(forward, dtype=float)

    f = forward.copy()
    f[2] = 0.0
    if np.linalg.norm(f) < 1e-8:
        f = np.array([1.0, 0.0, 0.0], dtype=float)

    f = _normalize(f)
    right = np.array([f[1], -f[0], 0.0], dtype=float)
    right = _normalize(right)

    half_w = np.tan(hfov * 0.5) * depth
    far_center = pos + f * depth

    p0 = pos[:2]
    p1 = (far_center + right * half_w)[:2]
    p2 = (far_center - right * half_w)[:2]
    return np.vstack([p0, p1, p2])


def _heading_angle_deg(forward):
    f = np.asarray(forward, dtype=float).copy()
    f[2] = 0.0
    if np.linalg.norm(f) < 1e-8:
        f = np.array([1.0, 0.0, 0.0], dtype=float)
    f = _normalize(f)
    return np.degrees(np.arctan2(f[1], f[0]))


def _drone_segments_2d(pos, forward, size, rotor_radius=None, n_circle_pts=28):
    pos = np.asarray(pos, dtype=float)
    forward = np.asarray(forward, dtype=float)

    f = forward.copy()
    f[2] = 0.0
    if np.linalg.norm(f) < 1e-8:
        f = np.array([1.0, 0.0, 0.0], dtype=float)
    f = _normalize(f)

    right = np.array([f[1], -f[0], 0.0], dtype=float)
    right = _normalize(right)

    d1 = _normalize(f + right)
    d2 = _normalize(f - right)

    p1 = pos + d1 * size
    p2 = pos - d1 * size
    p3 = pos + d2 * size
    p4 = pos - d2 * size

    segs = [
        np.array([p1[:2], p2[:2]], dtype=float),
        np.array([p3[:2], p4[:2]], dtype=float),
    ]

    if rotor_radius is None:
        rotor_radius = 0.22 * size

    t = np.linspace(0.0, 2.0 * np.pi, n_circle_pts)
    circle = np.stack(
        [rotor_radius * np.cos(t), rotor_radius * np.sin(t)],
        axis=1,
    )

    centers = [p1[:2], p2[:2], p3[:2], p4[:2]]
    for c in centers:
        segs.append(c + circle)

    return segs


class PaperViewer:
    def __init__(self, config, run_name="run", seed=99):
        self.config = config
        self.dt = float(config["dt"])
        self.interval_ms = int(self.dt * 1000)

        self.video_speedup = 4.0
        self.fps = max(1, int(round(self.video_speedup / self.dt)))

        self.run_name = run_name
        self.seed = seed

        self.drone_cfg = config["drone"]
        self.frames = []
        self.recording = False

        self.reward_hist = []
        self.monitor_hist = []
        self.disturb_hist = []

        self.output_dir = None
        self.floor_size = 500.0

        self.drone_trail_color = "tab:cyan"
        self.animal_trail_color = "tab:pink"

        self.drone_trail_style = dict(
            lw=2.0,
            alpha=0.60,
            linestyle="-",
        )
        self.animal_trail_style = dict(
            lw=2.0,
            alpha=0.60,
            linestyle="--",
        )

        self.animal_size = 12.5
        self.drone_size = 16.0
        self.entity_scale = 1.0

        self.video_camera_elev = 32
        self.video_camera_azim = -58
        self.video_z_margin = 20.0

        self.bg_color = "#fafafa"
        self.grid_color_2d = "#cfcfcf"
        self.grid_color_3d = "#d5d5d5"
        self.axis_color = "#555555"

    def _grid_step_from_span(self, span):
        if span <= 120:
            return 10
        elif span <= 300:
            return 25
        elif span <= 700:
            return 50
        else:
            return 100

    def _draw_xy_ground_grid_3d(self, ax):
        xmin, xmax, ymin, ymax = self._arena_bounds()
        zmin, _ = self._z_bounds()

        z0 = max(0.0, zmin)

        span = max(xmax - xmin, ymax - ymin)
        step = self._grid_step_from_span(span)

        xticks = np.arange(
            np.floor(xmin / step) * step,
            np.ceil(xmax / step) * step + step,
            step,
        )

        yticks = np.arange(
            np.floor(ymin / step) * step,
            np.ceil(ymax / step) * step + step,
            step,
        )

        for x in xticks:
            ax.plot(
                [x, x],
                [ymin, ymax],
                [z0, z0],
                color=self.grid_color_3d,
                lw=1.2,
                alpha=1.0,
                zorder=0,
            )

        for y in yticks:
            ax.plot(
                [xmin, xmax],
                [y, y],
                [z0, z0],
                color=self.grid_color_3d,
                lw=1.2,
                alpha=1.0,
                zorder=0,
            )

        return step, z0

    def _draw_origin_triad_2d(self, ax, step):
        x0, y0 = 0.0, 0.0
        triad_color = self.axis_color

        ax.annotate(
            "",
            xy=(x0 + step, y0),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.3, color=triad_color),
            zorder=2.2,
        )
        ax.annotate(
            "",
            xy=(x0, y0 + step),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.3, color=triad_color),
            zorder=2.2,
        )

        ax.text(x0 + 1.08 * step, y0 + 0.05 * step, "x", fontsize=11, color=triad_color, zorder=2.3)
        ax.text(x0 + 0.05 * step, y0 + 1.08 * step, "y", fontsize=11, color=triad_color, zorder=2.3)

        z_r = 0.12 * step
        ax.add_patch(
            Circle(
                (x0, y0),
                radius=z_r,
                facecolor="none",
                edgecolor=triad_color,
                lw=1.2,
                zorder=2.25,
            )
        )
        ax.add_patch(
            Circle(
                (x0, y0),
                radius=0.28 * z_r,
                facecolor=triad_color,
                edgecolor="none",
                zorder=2.3,
            )
        )
        ax.text(x0 + 0.32 * step, y0 + 0.32 * step, "z", fontsize=11, color=triad_color, zorder=2.3)

    def _draw_origin_triad_3d(self, ax, step, z0):
        color = self.axis_color
        arrow_ratio = 0.16

        ax.quiver(
            0.0, 0.0, z0,
            step, 0.0, 0.0,
            arrow_length_ratio=arrow_ratio,
            color=color,
            linewidth=1.3,
        )
        ax.quiver(
            0.0, 0.0, z0,
            0.0, step, 0.0,
            arrow_length_ratio=arrow_ratio,
            color=color,
            linewidth=1.3,
        )

        ax.text(1.08 * step, 0.03 * step, z0, "x", color=color, fontsize=10)
        ax.text(0.03 * step, 1.08 * step, z0, "y", color=color, fontsize=10)

        z_r = 0.12 * step
        ring = self._circle_points_3d(
            center=np.array([0.0, 0.0, z0], dtype=float),
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            radius=z_r,
            n=40,
        )
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, lw=1.2, alpha=1.0)
        ax.scatter([0.0], [0.0], [z0], s=14, c=color, depthshade=False)
        ax.text(0.32 * step, 0.32 * step, z0, "z", color=color, fontsize=10)

    def set_output_dir(self, path):
        self.output_dir = Path(path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_step_metrics(self, reward, info):
        self.reward_hist.append(float(reward))

        monitor_reward = _extract_metric(
            info,
            ["monitor_reward", "r_monitoring", "reward_monitoring"],
            default=0.0,
        )
        disturbance = _extract_metric(
            info,
            [
                "disturbance_penalty",
                "p_disturbance",
                "mean_disturbance",
                "disturbance",
                "cost_disturbance",
            ],
            default=0.0,
        )

        self.monitor_hist.append(monitor_reward)
        self.disturb_hist.append(disturbance)

    def _start_recording(self):
        self.frames = []
        self.recording = True
        print("[PaperViewer] Recording frames...")

    def draw(self, drones, animals, mode, fov=None, reward=None):
        if mode is None:
            if self.recording:
                self.close()
            return

        if not self.recording:
            self._start_recording()

        visible = np.zeros(len(animals), dtype=bool)

        if fov is not None and len(animals) > 0:
            try:
                animal_obs = fov[:, 4:].reshape(
                    len(drones),
                    len(animals),
                    self.config.model.space.animal_features,
                )
                visible = np.any(animal_obs[:, :, 0] == 1.0, axis=0)
            except Exception:
                pass

        self.frames.append(
            dict(
                drones=np.array([d.pos.to_numpy() for d in drones], dtype=float),
                drone_types=[d.drone_type for d in drones],
                animals=np.array([a.pos.to_numpy() for a in animals], dtype=float),
                views=np.array([d.view_dir.to_numpy() for d in drones], dtype=float),
                visible=visible,
            )
        )

    def _make_output_dir_if_needed(self):
        if self.output_dir is not None:
            return

        repo_root = Path(__file__).resolve().parent.parent
        recordings_dir = repo_root / "recordings"
        recordings_dir.mkdir(exist_ok=True)

        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"{self.run_name}_seed{self.seed}_{stamp}"
        self.output_dir = recordings_dir / folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _arena_bounds(self):
        if not self.frames:
            half = 0.5 * self.floor_size
            return -half, half, -half, half

        all_xy = []

        for frame in self.frames:
            if len(frame["drones"]) > 0:
                all_xy.append(frame["drones"][:, :2])
            if len(frame["animals"]) > 0:
                all_xy.append(frame["animals"][:, :2])

        if not all_xy:
            half = 0.5 * self.floor_size
            return -half, half, -half, half

        pts = np.vstack(all_xy)
        xmin = float(np.min(pts[:, 0]))
        xmax = float(np.max(pts[:, 0]))
        ymin = float(np.min(pts[:, 1]))
        ymax = float(np.max(pts[:, 1]))

        xspan = max(xmax - xmin, 1.0)
        yspan = max(ymax - ymin, 1.0)

        xpad = max(0.03 * xspan, 8.0)
        ypad = max(0.03 * yspan, 8.0)

        xmin -= xpad
        xmax += xpad
        ymin -= ypad
        ymax += ypad

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * max(xmax - xmin, ymax - ymin)

        return cx - half, cx + half, cy - half, cy + half

    def _z_bounds(self):
        if not self.frames:
            return 0.0, 150.0

        zs = []
        for frame in self.frames:
            if len(frame["drones"]) > 0:
                zs.append(frame["drones"][:, 2])
            if len(frame["animals"]) > 0:
                zs.append(frame["animals"][:, 2])

        if not zs:
            return 0.0, 150.0

        z = np.concatenate(zs)
        zmin = float(np.min(z))
        zmax = float(np.max(z))

        zmin = min(0.0, zmin - self.video_z_margin)
        zmax = max(10.0, zmax + self.video_z_margin)

        if zmax - zmin < 50.0:
            zmax = zmin + 50.0

        return zmin, zmax

    def _setup_axes(self, ax):
        xmin, xmax, ymin, ymax = self._arena_bounds()

        fig = ax.figure
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        for spine in ax.spines.values():
            spine.set_visible(False)

        span = max(xmax - xmin, ymax - ymin)
        step = self._grid_step_from_span(span)

        xticks = np.arange(
            np.floor(xmin / step) * step,
            np.ceil(xmax / step) * step + step,
            step,
        )
        yticks = np.arange(
            np.floor(ymin / step) * step,
            np.ceil(ymax / step) * step + step,
            step,
        )

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            labelbottom=False,
            labelleft=False,
        )

        ax.grid(
            True,
            which="major",
            color=self.grid_color_2d,
            linestyle="-",
            linewidth=1.25,
            alpha=1.0,
            zorder=0,
        )

        self._draw_origin_triad_2d(ax, step)

    def _setup_axes_3d(self, ax):
        xmin, xmax, ymin, ymax = self._arena_bounds()
        zmin, zmax = self._z_bounds()

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        xspan = max(xmax - xmin, 1.0)
        yspan = max(ymax - ymin, 1.0)
        zspan = max(zmax - zmin, 1.0)

        ax.set_box_aspect((xspan, yspan, 0.42 * zspan))
        ax.view_init(elev=self.video_camera_elev, azim=self.video_camera_azim)

        try:
            ax.set_proj_type("persp", focal_length=1.35)
        except Exception:
            pass

        fig = ax.figure
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        except Exception:
            pass

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        try:
            ax.xaxis.line.set_color((1, 1, 1, 0))
            ax.yaxis.line.set_color((1, 1, 1, 0))
            ax.zaxis.line.set_color((1, 1, 1, 0))
        except Exception:
            pass

        step, z0 = self._draw_xy_ground_grid_3d(ax)
        self._draw_origin_triad_3d(ax, step, z0)

    def _save_snapshot_3d(self, frame_idx, filename):
        fig = plt.figure(figsize=(7.2, 7.2), dpi=220)
        fig.patch.set_facecolor(self.bg_color)

        ax = fig.add_subplot(111, projection="3d")
        self._setup_axes_3d(ax)

        frame = self.frames[frame_idx]
        d = frame["drones"]
        a = frame["animals"]
        v = frame["views"]
        visible = frame["visible"]

        for j in range(len(d)):
            pts = np.array([f["drones"][j] for f in self.frames[:frame_idx + 1]])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=self.drone_trail_color,
                zorder=7,
                **self.drone_trail_style,
            )

        for j in range(len(a)):
            pts = np.array([f["animals"][j] for f in self.frames[:frame_idx + 1]])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=self.animal_trail_color,
                zorder=4,
                **self.animal_trail_style,
            )

        for j in range(len(a)):
            motion = self._get_motion(self.frames, frame_idx, j, kind="animals")
            self._draw_animal_3d(ax, a[j], motion, bool(visible[j]))

        for j in range(len(d)):
            self._draw_drone_3d(ax, d[j], v[j], frame["drone_types"][j], self.drone_size)

        fig.savefig(
            self.output_dir / filename,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

    def _draw_soft_shadow(self, ax, x, y, angle_deg, width, height, alpha=0.12, dx=6.0, dy=-5.0):
        shadow = Ellipse(
            (x + dx, y + dy),
            width=width,
            height=height,
            angle=angle_deg,
            facecolor="black",
            edgecolor="none",
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(shadow)

    def _draw_animal(self, ax, pos, forward, visible, size):
        x, y = pos[:2]
        angle = _heading_angle_deg(forward)

        scaled_size = self.entity_scale * size
        body_w = 2.2 * scaled_size
        body_h = 1.15 * scaled_size

        self._draw_soft_shadow(
            ax,
            x,
            y,
            angle,
            1.08 * body_w,
            0.95 * body_h,
            alpha=0.18,
            dx=4.5,
            dy=-3.5,
        )

        body = Ellipse(
            (x, y),
            width=body_w,
            height=body_h,
            angle=angle,
            facecolor="#d3bc4e" if visible else "#8a5f58",
            edgecolor="#d55958",
            linewidth=1.7,
            alpha=0.98,
            zorder=6,
        )
        ax.add_patch(body)

    def _draw_drone(self, ax, pos, forward, drone_type, size):
        x, y = pos[:2]
        f = np.asarray(forward, dtype=float).copy()
        f[2] = 0.0
        if np.linalg.norm(f) < 1e-8:
            f = np.array([1.0, 0.0, 0.0], dtype=float)
        f = _normalize(f)

        scaled_size = self.entity_scale * size

        cfg = self.drone_cfg[drone_type]
        depth = float(cfg["view_range"]) / 6.0
        hfov = np.deg2rad(float(cfg["hor_angle"]))
        poly = _frustum_ground_polygon(pos, f, hfov, depth)

        frustum_fill = Polygon(
            poly,
            closed=True,
            facecolor="black",
            edgecolor="none",
            alpha=0.03,
            zorder=8,
        )
        ax.add_patch(frustum_fill)

        frustum_edge = Polygon(
            poly,
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.18,
            zorder=8.2,
        )
        ax.add_patch(frustum_edge)

        angle = _heading_angle_deg(f)
        shadow = Ellipse(
            (x + 6.0, y - 6.0),
            width=1.9 * scaled_size,
            height=1.2 * scaled_size,
            angle=angle,
            facecolor="black",
            edgecolor="none",
            alpha=0.08,
            zorder=8.5,
        )
        ax.add_patch(shadow)

        segments = _drone_segments_2d(
            pos,
            f,
            scaled_size,
            rotor_radius=0.22 * scaled_size,
            n_circle_pts=28,
        )

        for seg in segments[:2]:
            ax.plot(seg[:, 0], seg[:, 1], color="#222222", lw=1.8, alpha=0.96, zorder=9)

        for seg in segments[2:]:
            ax.plot(seg[:, 0], seg[:, 1], color="#222222", lw=1.1, alpha=0.96, zorder=9)

    def _ellipse_points_3d(self, center, forward, width, height, n=40):
        center = np.asarray(center, dtype=float)

        f = np.asarray(forward, dtype=float).copy()
        f[2] = 0.0
        if np.linalg.norm(f) < 1e-8:
            f = np.array([1.0, 0.0, 0.0], dtype=float)
        f = _normalize(f)

        right = np.array([f[1], -f[0], 0.0], dtype=float)
        right = _normalize(right)

        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

        pts = []
        for tt in t:
            p = center + 0.5 * width * np.cos(tt) * f + 0.5 * height * np.sin(tt) * right
            pts.append(p)

        return np.asarray(pts, dtype=float)

    def _circle_points_3d(self, center, normal, radius, n=40):
        center = np.asarray(center, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal = _normalize(normal)

        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(normal, ref)) > 0.95:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)

        u = np.cross(normal, ref)
        u = _normalize(u)
        v = np.cross(normal, u)
        v = _normalize(v)

        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
        pts = []
        for tt in t:
            p = center + radius * (np.cos(tt) * u + np.sin(tt) * v)
            pts.append(p)

        return np.asarray(pts, dtype=float)

    def _frustum_corners_3d(self, pos, forward, up, hfov, vfov, depth):
        pos = np.asarray(pos, dtype=float)
        forward = np.asarray(forward, dtype=float)
        up = np.asarray(up, dtype=float)

        f = _normalize(forward)
        u = _normalize(up)

        right = np.cross(f, u)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        right = _normalize(right)

        u = np.cross(right, f)
        u = _normalize(u)

        half_w = np.tan(hfov * 0.5) * depth
        half_h = np.tan(vfov * 0.5) * depth

        far_center = pos + f * depth

        far_tl = far_center - right * half_w + u * half_h
        far_tr = far_center + right * half_w + u * half_h
        far_bl = far_center - right * half_w - u * half_h
        far_br = far_center + right * half_w - u * half_h

        return far_tl, far_tr, far_bl, far_br

    def _draw_rotor_ring_3d(self, ax, center, normal, radius, color="#222222", lw=1.1, zorder=9):
        pts = self._circle_points_3d(center, normal, radius, n=44)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            lw=lw,
            alpha=0.96,
            zorder=zorder,
        )

    def _draw_animal_3d(self, ax, pos, forward, visible):
        pos = np.asarray(pos, dtype=float).copy()

        x, y, z = pos
        scaled_size = self.entity_scale * self.animal_size
        body_w = 2.2 * scaled_size
        body_h = 1.15 * scaled_size

        face = "#d3bc4e" if visible else "#8a5f58"
        edge = "#d55958"

        shadow_center = np.array([x + 4.5, y - 3.5, 0.0], dtype=float)
        shadow_pts = self._ellipse_points_3d(
            shadow_center,
            forward,
            width=1.08 * body_w,
            height=0.95 * body_h,
            n=40,
        )
        shadow = Poly3DCollection(
            [shadow_pts],
            facecolors="black",
            edgecolors="none",
            alpha=0.10,
            zorder=3,
        )
        ax.add_collection3d(shadow)

        body_pts = self._ellipse_points_3d(
            np.array([x, y, z], dtype=float),
            forward,
            width=body_w,
            height=body_h,
            n=40,
        )
        body = Poly3DCollection(
            [body_pts],
            facecolors=face,
            edgecolors=edge,
            linewidths=1.1,
            alpha=0.98,
            zorder=5,
        )
        ax.add_collection3d(body)

    def _draw_drone_3d(self, ax, pos, forward, drone_type, size):
        pos = np.asarray(pos, dtype=float)

        f = np.asarray(forward, dtype=float).copy()
        if np.linalg.norm(f) < 1e-8:
            f = np.array([1.0, 0.0, 0.0], dtype=float)
        f = _normalize(f)

        scaled_size = self.entity_scale * size

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)

        right = np.cross(f, world_up)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        right = _normalize(right)

        up = np.cross(right, f)
        up = _normalize(up)

        d1 = _normalize(f + right)
        d2 = _normalize(f - right)

        p1 = pos + d1 * scaled_size
        p2 = pos - d1 * scaled_size
        p3 = pos + d2 * scaled_size
        p4 = pos - d2 * scaled_size

        shadow_pts = self._ellipse_points_3d(
            np.array([pos[0] + 6.0, pos[1] - 6.0, 0.0], dtype=float),
            f,
            width=1.9 * scaled_size,
            height=1.2 * scaled_size,
            n=40,
        )
        shadow = Poly3DCollection(
            [shadow_pts],
            facecolors="black",
            edgecolors="none",
            alpha=0.08,
            zorder=8,
        )
        ax.add_collection3d(shadow)

        for a, b in [(p1, p2), (p3, p4)]:
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                lw=1.8,
                alpha=0.96,
                color="#222222",
                zorder=9,
            )

        ax.scatter(
            [pos[0]],
            [pos[1]],
            [pos[2]],
            s=18,
            c="#222222",
            alpha=0.98,
            depthshade=False,
            zorder=9,
        )

        rotor_radius = 0.22 * scaled_size
        rotor_normal = up

        for p in [p1, p2, p3, p4]:
            self._draw_rotor_ring_3d(
                ax,
                center=p,
                normal=rotor_normal,
                radius=rotor_radius,
                color="#222222",
                lw=1.15,
                zorder=9,
            )
            ax.scatter(
                [p[0]],
                [p[1]],
                [p[2]],
                s=7,
                c="#222222",
                alpha=0.95,
                depthshade=False,
                zorder=9,
            )

        cfg = self.drone_cfg[drone_type]
        depth = float(cfg["view_range"]) / 6.0
        hfov = np.deg2rad(float(cfg["hor_angle"]))
        vfov = np.deg2rad(float(cfg["ver_angle"])) if "ver_angle" in cfg else np.deg2rad(60.0)

        far_tl, far_tr, far_bl, far_br = self._frustum_corners_3d(
            pos=pos,
            forward=f,
            up=up,
            hfov=hfov,
            vfov=vfov,
            depth=depth,
        )

        frustum_edges = [
            (pos, far_tl),
            (pos, far_tr),
            (pos, far_bl),
            (pos, far_br),
            (far_tl, far_tr),
            (far_tr, far_br),
            (far_br, far_bl),
            (far_bl, far_tl),
        ]

        for a, b in frustum_edges:
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                [a[2], b[2]],
                lw=0.9,
                alpha=0.18,
                color="black",
                zorder=8.5,
            )

        far_face = Poly3DCollection(
            [[far_tl, far_tr, far_br, far_bl]],
            facecolors="black",
            edgecolors="none",
            alpha=0.025,
            zorder=8.2,
        )
        ax.add_collection3d(far_face)

    def _get_motion(self, frames, i, j, kind="animals"):
        arr = frames[i][kind]
        if i > 0:
            motion = arr[j] - frames[i - 1][kind][j]
        elif len(frames) > 1:
            motion = frames[1][kind][j] - arr[j]
        else:
            motion = np.array([1.0, 0.0, 0.0], dtype=float)

        if np.linalg.norm(motion[:2]) < 1e-8:
            motion = np.array([1.0, 0.0, 0.0], dtype=float)
        return motion

    def _save_snapshot(self, frame_idx, filename):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=220)
        self._setup_axes(ax)

        frame = self.frames[frame_idx]
        d = frame["drones"]
        a = frame["animals"]
        v = frame["views"]
        visible = frame["visible"]

        for j in range(len(d)):
            pts = np.array([f["drones"][j][:2] for f in self.frames[: frame_idx + 1]])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=self.drone_trail_color,
                zorder=3,
                **self.drone_trail_style,
            )

        for j in range(len(a)):
            pts = np.array([f["animals"][j][:2] for f in self.frames[: frame_idx + 1]])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=self.animal_trail_color,
                zorder=2.8,
                **self.animal_trail_style,
            )

        for j in range(len(a)):
            motion = self._get_motion(self.frames, frame_idx, j, kind="animals")
            self._draw_animal(ax, a[j], motion, bool(visible[j]), self.animal_size)

        for j in range(len(d)):
            self._draw_drone(ax, d[j], v[j], frame["drone_types"][j], self.drone_size)

        fig.savefig(self.output_dir / filename, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_trajectories(self):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=240)
        self._setup_axes(ax)

        n_drones = len(self.frames[0]["drones"])
        n_animals = len(self.frames[0]["animals"])

        for j in range(n_drones):
            pts = np.array([f["drones"][j][:2] for f in self.frames])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=self.drone_trail_color,
                zorder=3,
                **self.drone_trail_style,
            )

        for j in range(n_animals):
            pts = np.array([f["animals"][j][:2] for f in self.frames])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=self.animal_trail_color,
                zorder=2.8,
                **self.animal_trail_style,
            )

        fig.savefig(self.output_dir / "trajectories.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_reward_plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), dpi=220)

        x = np.arange(len(self.reward_hist))
        series = [
            (self.reward_hist, "Final reward"),
            (self.monitor_hist, "Monitor reward"),
            (self.disturb_hist, "Disturbance penalty"),
        ]

        for ax, (y, title) in zip(axes, series):
            y = np.asarray(y, dtype=float)
            ax.plot(x, y, lw=0.3, alpha=0.9, color="black")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.10)
            ax.set_xlabel("Step", fontsize=9)
            ax.tick_params(labelsize=8)

        axes[0].set_ylabel("Value", fontsize=9)

        fig.tight_layout()
        fig.savefig(self.output_dir / "rewards.png", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

    def _save_video(self):
        fig = plt.figure(figsize=(7.2, 7.2), dpi=180)
        fig.patch.set_facecolor(self.bg_color)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.0, 0.0, 1.0, 1.0])

        def update(i):
            ax.cla()
            self._setup_axes_3d(ax)

            frame = self.frames[i]
            d = frame["drones"]
            a = frame["animals"]
            v = frame["views"]
            visible = frame["visible"]

            for j in range(len(d)):
                pts = np.array([f["drones"][j] for f in self.frames[: i + 1]])
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=self.drone_trail_color,
                    zorder=7,
                    **self.drone_trail_style,
                )

            for j in range(len(a)):
                pts = np.array([f["animals"][j] for f in self.frames[: i + 1]])
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=self.animal_trail_color,
                    zorder=4,
                    **self.animal_trail_style,
                )

            for j in range(len(a)):
                motion = self._get_motion(self.frames, i, j, kind="animals")
                self._draw_animal_3d(ax, a[j], motion, bool(visible[j]))

            for j in range(len(d)):
                self._draw_drone_3d(ax, d[j], v[j], frame["drone_types"][j], self.drone_size)

            return []

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.frames),
            interval=1,
            blit=False,
        )

        anim.save(
            self.output_dir / "video.mp4",
            writer=FFMpegWriter(fps=self.fps),
        )

        plt.close(fig)

    def close(self):
        if not self.frames:
            return

        self._make_output_dir_if_needed()

        print(f"[PaperViewer] Rendering outputs to: {self.output_dir}")

        self._save_video()

        n = len(self.frames)

        progress_points = [
            (0.25, "snapshot25_2d.png", "snapshot25_3d.png"),
            (0.50, "snapshot50_2d.png", "snapshot50_3d.png"),
            (0.75, "snapshot75_2d.png", "snapshot75_3d.png"),
            (1.00, "snapshot100_2d.png", "snapshot100_3d.png"),
        ]

        for frac, name2d, name3d in progress_points:
            idx = min(n - 1, max(0, int(np.floor(frac * (n - 1)))))
            self._save_snapshot(idx, name2d)
            self._save_snapshot_3d(idx, name3d)

        self._save_trajectories()
        self._save_reward_plot()

        print("[PaperViewer] Done.")
        self.recording = False
        self.frames = []


def init_agent(config, run_dir, weight_type="best"):
    agent_type = config.agent_type

    if agent_type == "ppo":
        agent = PPOAgent(config)
    elif agent_type == "mappo":
        agent = MAPPOAgent(config)
    elif agent_type == "sac":
        agent = SACAgent(config)
    elif agent_type == "dqn":
        agent = DQNAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    if agent_type == "ppo":
        agent.actor.chkpt_dir = run_dir
        agent.critic.chkpt_dir = run_dir
    elif agent_type == "mappo":
        agent.actor.chkpt_dir = run_dir
        agent.critic.chkpt_dir = run_dir
    elif agent_type == "sac":
        agent.actor.chkpt_dir = run_dir
        agent.critic_1.chkpt_dir = run_dir
        agent.critic_2.chkpt_dir = run_dir
        agent.target_critic_1.chkpt_dir = run_dir
        agent.target_critic_2.chkpt_dir = run_dir
    elif agent_type == "dqn":
        agent.q_net.chkpt_dir = run_dir
        agent.target_q_net.chkpt_dir = run_dir

    agent.load_models(name=weight_type)

    if agent_type == "ppo":
        agent.actor.eval()
        agent.critic.eval()
    elif agent_type == "mappo":
        agent.actor.eval()
        agent.critic.eval()
    elif agent_type == "sac":
        agent.actor.eval()
        agent.critic_1.eval()
        agent.critic_2.eval()
        agent.target_critic_1.eval()
        agent.target_critic_2.eval()
    elif agent_type == "dqn":
        agent.q_net.eval()
        agent.target_q_net.eval()

    return agent, agent_type


def choose_action(agent, obs, agent_type):
    if agent_type == "ppo":
        actions = []
        for drone_obs in obs:
            action, _, _ = agent.choose_action(drone_obs, deterministic=True)
            actions.append(action)
        return np.array(actions, dtype=np.float32)

    elif agent_type == "mappo":
        with torch.no_grad():
            actions, _, _ = agent.choose_actions(obs, deterministic=True)
        return np.asarray(actions, dtype=np.float32)

    elif agent_type == "sac":
        obs_arr = np.asarray(obs, dtype=np.float32)
        joint_obs = obs_arr.reshape(-1)
        joint_action_flat, _, _ = agent.choose_action(joint_obs, deterministic=True)
        env_action = np.asarray(joint_action_flat, dtype=np.float32).reshape(obs_arr.shape[0], -1)
        return env_action

    elif agent_type == "dqn":
        obs_arr = np.asarray(obs, dtype=np.float32)

        if obs_arr.shape[0] != 1:
            raise ValueError(
                "DQN playback currently expects exactly 1 drone, "
                f"but got obs shape {obs_arr.shape}"
            )

        single_obs = obs_arr[0]
        action_vec, _, _ = agent.choose_action(single_obs, deterministic=True)
        env_action = np.asarray(action_vec, dtype=np.float32).reshape(1, -1)
        return env_action

    else:
        raise ValueError(agent_type)


def main(config, run_dir, seed, model_type="best"):
    config = Box(config)

    env = Env(config, render_mode="human", seed=seed)
    obs, info = env.reset()

    run_name = Path(run_dir).name
    paper_viewer = PaperViewer(config, run_name=run_name, seed=seed)

    repo_root = Path(__file__).resolve().parent.parent
    recordings_dir = repo_root / "recordings"
    recordings_dir.mkdir(exist_ok=True)

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = recordings_dir / f"{run_name}_seed{seed}_{stamp}"
    paper_viewer.set_output_dir(out_dir)

    env.viewer = paper_viewer

    agent, agent_type = init_agent(config, run_dir, model_type)
    print(f"Loaded agent type: {agent_type}")

    terminated = False
    truncated = False
    step_count = 0
    episode_reward = 0.0

    while not (terminated or truncated):
        with torch.no_grad():
            action = choose_action(agent, obs, agent_type)

        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        episode_reward += float(reward)
        env.viewer.log_step_metrics(reward, info)

    norm_reward = episode_reward / config.max_episode_steps

    print(f"Episode finished in {step_count} steps")
    print(f"Normalized Reward: {norm_reward:.4f}")
    print(f"Artifacts saved in: {out_dir}")

    env.viewer.close()


def _init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run folder name or 'latest'",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="Random seed for environment",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="best",
        help="best or latest weights",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _init_argparse()
    cfg, run_dir = load_run(args.run)
    main(cfg, run_dir, seed=args.seed, model_type=args.weights)