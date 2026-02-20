import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def set_axes_equal(ax, xlim, ylim, zlim):
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])

    r = 0.5 * max(
        abs(xlim[1] - xlim[0]),
        abs(ylim[1] - ylim[0]),
        abs(zlim[1] - zlim[0]),
    )

    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)


def _frustum_segments(pos, view_dir, hfov, vfov, depth):
    view_dir = np.asarray(view_dir, dtype=float)
    n = np.linalg.norm(view_dir)
    if n < 1e-8:
        return []

    forward = view_dir / n
    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-8 else np.array([1.0, 0.0, 0.0])

    up = np.cross(right, forward)
    un = np.linalg.norm(up)
    if un < 1e-8:
        return []
    up /= un

    tan_h = np.tan(hfov * 0.5)
    tan_v = np.tan(vfov * 0.5)

    center = pos + forward * depth

    corners = [
        center + right * depth * tan_h + up * depth * tan_v,
        center - right * depth * tan_h + up * depth * tan_v,
        center - right * depth * tan_h - up * depth * tan_v,
        center + right * depth * tan_h - up * depth * tan_v,
    ]

    segs = []
    for c in corners:
        segs.append(((pos[0], c[0]), (pos[1], c[1]), (pos[2], c[2])))

    loop = corners + [corners[0]]
    for a, b in zip(loop[:-1], loop[1:]):
        segs.append(((a[0], b[0]), (a[1], b[1]), (a[2], b[2])))

    return segs


def animate_from_log(
    log,
    save_path=None,
    hfov_deg=90,
    vfov_deg=56,
    frustum_depth=15,
    title="Simulation Episode",
    interval_ms=200,
    trail=20,
    playback_speed=4.0,
    show_frustum=True,
    ortho=True,
):
    # ---- Organize log by time ----
    times = sorted({row["t"] for row in log})
    frames = []
    for t in times:
        frames.append([r for r in log if r["t"] == t])

    agent_ids = sorted({r["agent_id"] for r in log})
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    species = {}
    for r in log:
        species[r["agent_id"]] = r["type"]

    nF = len(frames)
    nA = len(agent_ids)

    pos = np.full((nF, nA, 3), np.nan)
    view = np.full((nF, nA, 3), np.nan)

    for fi, rows in enumerate(frames):
        for r in rows:
            ai = id_to_idx[r["agent_id"]]
            pos[fi, ai] = [r["x"], r["y"], r["z"]]

            if "view_x" in r:
                view[fi, ai] = [r["view_x"], r["view_y"], r["view_z"]]

    # ---- Axis limits ----
    valid = ~np.isnan(pos[..., 0])
    xs = pos[..., 0][valid]
    ys = pos[..., 1][valid]
    zs = pos[..., 2][valid]

    xlim = (xs.min(), xs.max())
    ylim = (ys.min(), ys.max())
    zlim = (zs.min(), zs.max())

    # ---- Colors ----
    species_colors = {
        "standard_drone": "blue",
        "standard_animal": "orange",
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    if ortho:
        ax.set_proj_type("ortho")

    ax.view_init(elev=30, azim=-60)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title} | playback x{playback_speed:.2f}")

    set_axes_equal(ax, xlim, ylim, zlim)

    traj_lines = []
    curr_pts = []

    for aid in agent_ids:
        sp = species[aid]
        color = species_colors.get(sp, "black")
        lw = 2.5 if sp == "standard_drone" else 1.2
        alpha = 1.0 if sp == "standard_drone" else 0.7

        line, = ax.plot([], [], [], color=color, linewidth=lw, alpha=alpha)
        pt = ax.scatter([], [], [], color=color, s=30)

        traj_lines.append(line)
        curr_pts.append(pt)

    drone_idxs = [
        id_to_idx[aid]
        for aid in agent_ids
        if species[aid] == "standard_drone"
    ]

    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    frustum_lines = []
    if show_frustum:
        for _ in drone_idxs:
            drone_lines = []
            for _ in range(8):
                ln, = ax.plot([], [], [], color="blue", alpha=0.25)
                drone_lines.append(ln)
            frustum_lines.append(drone_lines)

    def update(frame_idx):
        start = max(0, frame_idx - trail)

        for ai in range(nA):
            p = pos[start:frame_idx + 1, ai]
            p = p[~np.isnan(p[:, 0])]

            if len(p) > 0:
                traj_lines[ai].set_data(p[:, 0], p[:, 1])
                traj_lines[ai].set_3d_properties(p[:, 2])
                curr_pts[ai]._offsets3d = ([p[-1, 0]], [p[-1, 1]], [p[-1, 2]])
            else:
                traj_lines[ai].set_data([], [])
                traj_lines[ai].set_3d_properties([])
                curr_pts[ai]._offsets3d = ([], [], [])

        if show_frustum:
            for d, di in enumerate(drone_idxs):
                p = pos[frame_idx, di]
                v = view[frame_idx, di]

                segs = []
                if not np.any(np.isnan(p)) and not np.any(np.isnan(v)):
                    segs = _frustum_segments(p, v, hfov, vfov, frustum_depth)

                for i, ln in enumerate(frustum_lines[d]):
                    if i < len(segs):
                        xs, ys, zs = segs[i]
                        ln.set_data(xs, ys)
                        ln.set_3d_properties(zs)
                    else:
                        ln.set_data([], [])
                        ln.set_3d_properties([])

        return traj_lines + curr_pts

    anim = FuncAnimation(
        fig,
        update,
        frames=nF,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    render_fps = (1000.0 / interval_ms) * playback_speed

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        anim.save(save_path, writer="ffmpeg", dpi=200, fps=render_fps)

    return fig, ax, anim