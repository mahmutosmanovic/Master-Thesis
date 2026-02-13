import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def set_axes_equal(ax, xlim, ylim, zlim):
    """Force equal data scaling using fixed limits (important for animation)."""
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])

    xr = abs(xlim[1] - xlim[0])
    yr = abs(ylim[1] - ylim[0])
    zr = abs(zlim[1] - zlim[0])

    r = 0.5 * max(xr, yr, zr)

    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)


def _frustum_segments(pos, view_dir, hfov, vfov, depth):
    """
    Return list of line segments for a frustum:
      - 4 rays: apex -> each far corner
      - 4 edges: far-plane rectangle
    Each segment is ((x0,x1),(y0,y1),(z0,z1))
    """
    view_dir = np.asarray(view_dir, dtype=float)
    n = np.linalg.norm(view_dir)
    if n < 1e-8:
        return []

    forward = view_dir / n
    world_up = np.array([0.0, 0.0, 1.0])

    # Right in XY plane (no roll)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn

    # Up completes basis
    up = np.cross(right, forward)
    un = np.linalg.norm(up)
    if un < 1e-8:
        return []
    up /= un

    tan_h = np.tan(hfov * 0.5)
    tan_v = np.tan(vfov * 0.5)

    center = pos + forward * depth

    corners = [
        center + right * (depth * tan_h) + up * (depth * tan_v),  # TR
        center - right * (depth * tan_h) + up * (depth * tan_v),  # TL
        center - right * (depth * tan_h) - up * (depth * tan_v),  # BL
        center + right * (depth * tan_h) - up * (depth * tan_v),  # BR
    ]

    segs = []
    # Rays
    for c in corners:
        segs.append(((pos[0], c[0]), (pos[1], c[1]), (pos[2], c[2])))

    # Far plane edges
    loop = corners + [corners[0]]
    for a, b in zip(loop[:-1], loop[1:]):
        segs.append(((a[0], b[0]), (a[1], b[1]), (a[2], b[2])))

    return segs


def animate_simulation_csv_3d(
    csv_path,
    hfov_deg=90.0,
    vfov_deg=60.0,
    frustum_depth=15.0,
    interval_ms=50,
    trail=800,              # how many past points to show per agent
    show_frustum=True,
    ortho=True,
):
    """
    Animate trajectories + drone frustum from your CSV log.

    Requirements:
      CSV contains columns: t, agent_id, species, x,y,z and for drones view_x,view_y,view_z.

    Notes:
      - Uses fixed axis limits + equal scaling for stable visuals.
      - Draws drone frustum at each frame (latest drone pose).
    """

    # --- Load all rows ---
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Parse to numeric arrays per frame. We assume rows are already in time order or have 't'.
    # We'll build frames grouped by t (string float).
    # If 't' missing in your CSV, you can group by row index instead.
    times = np.array([float(r.get("t", i)) for i, r in enumerate(rows)], dtype=float)

    # Unique times in sorted order
    unique_t = np.unique(times)

    # Build per-time list of rows (frame -> rows)
    frame_rows = []
    for t in unique_t:
        frame_rows.append([r for r, tt in zip(rows, times) if tt == t])

    # Collect all agent ids & species
    agent_ids = sorted({int(r["agent_id"]) for r in rows})
    agent_species = {}
    for r in rows:
        aid = int(r["agent_id"])
        if aid not in agent_species:
            agent_species[aid] = r["type"]

    # Precompute positions per agent per frame (NaN when not present)
    nF = len(frame_rows)
    nA = len(agent_ids)
    id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}

    pos = np.full((nF, nA, 3), np.nan, dtype=float)
    view = np.full((nF, nA, 3), np.nan, dtype=float)  # only meaningful for drones

    for fi, rows_at_t in enumerate(frame_rows):
        for r in rows_at_t:
            ai = id_to_idx[int(r["agent_id"])]
            pos[fi, ai, 0] = float(r["x"])
            pos[fi, ai, 1] = float(r["y"])
            pos[fi, ai, 2] = float(r["z"])
            # view may be empty for non-drones
            vx = r.get("view_x", "")
            vy = r.get("view_y", "")
            vz = r.get("view_z", "")
            if vx != "" and vy != "" and vz != "":
                view[fi, ai, 0] = float(vx)
                view[fi, ai, 1] = float(vy)
                view[fi, ai, 2] = float(vz)

    # Axis limits from all positions
    valid = ~np.isnan(pos[..., 0])
    xs = pos[..., 0][valid]
    ys = pos[..., 1][valid]
    zs = pos[..., 2][valid]
    xlim = (float(xs.min()), float(xs.max()))
    ylim = (float(ys.min()), float(ys.max()))
    zlim = (float(zs.min()), float(zs.max()))

    # Color map
    species_colors = {
        "jackal": "orange",
        "pigeon": "green",
        "eagle": "red",
        "drone": "blue",
    }

    # --- Setup figure ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    if ortho:
        ax.set_proj_type("ortho")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Simulation With 3 PPO Drones and 3 animals (rw, poi, path)")

    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)
    set_axes_equal(ax, xlim, ylim, zlim)

    traj_lines = []
    curr_pts = []

    for aid in agent_ids:
        sp = agent_species[aid]
        color = species_colors.get(sp, "black")
        lw = 2.5 if sp == "drone" else 1.2
        al = 1.0 if sp == "drone" else 0.7

        line, = ax.plot([], [], [], color=color, linewidth=lw, alpha=al)
        pt = ax.scatter([], [], [], color=color, s=25)

        traj_lines.append(line)
        curr_pts.append(pt)

    drone_idxs = [id_to_idx[aid] for aid in agent_ids if agent_species[aid] == "drone"]

    frustum_lines = []
    if show_frustum:
        for _ in drone_idxs:
            drone_lines = []
            for _ in range(8):  # 8 segments per frustum
                ln, = ax.plot(
                    [], [], [],
                    color=species_colors.get("drone", "blue"),
                    alpha=0.25,
                    linewidth=1.2
                )
                drone_lines.append(ln)
            frustum_lines.append(drone_lines)

    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    def init():
        for line in traj_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for pt in curr_pts:
            pt._offsets3d = ([], [], [])
        for drone_lines in frustum_lines:
            for ln in drone_lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
        return traj_lines + curr_pts + frustum_lines

    def update(frame_idx):
        # Update trajectories and current points
        start = max(0, frame_idx - trail)
        for ai, aid in enumerate(agent_ids):
            p = pos[start:frame_idx+1, ai, :]  # (T,3)
            m = ~np.isnan(p[:, 0])
            p = p[m]
            if p.shape[0] > 0:
                traj_lines[ai].set_data(p[:, 0], p[:, 1])
                traj_lines[ai].set_3d_properties(p[:, 2])

                curr = p[-1]
                curr_pts[ai]._offsets3d = ([curr[0]], [curr[1]], [curr[2]])
            else:
                traj_lines[ai].set_data([], [])
                traj_lines[ai].set_3d_properties([])
                curr_pts[ai]._offsets3d = ([], [], [])

        if show_frustum:
            for d, di in enumerate(drone_idxs):
                p = pos[frame_idx, di, :]
                v = view[frame_idx, di, :]

                segs = []
                if not (np.any(np.isnan(p)) or np.any(np.isnan(v))):
                    segs = _frustum_segments(p, v, hfov, vfov, frustum_depth)

                drone_lines = frustum_lines[d]
                for i, ln in enumerate(drone_lines):
                    if i < len(segs):
                        xs, ys, zs = segs[i]
                        ln.set_data(xs, ys)
                        ln.set_3d_properties(zs)
                    else:
                        ln.set_data([], [])
                        ln.set_3d_properties([])

        return traj_lines + curr_pts + [ln for drone in frustum_lines for ln in drone]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(unique_t),
        init_func=init,
        interval=interval_ms,
        blit=False,  # mplot3d doesn't blit reliably
        repeat=True,
    )

    plt.show()
    return fig, ax, anim


if __name__ == "__main__":
    animate_simulation_csv_3d(
        "logs/simulations/behaviour_test.csv",
        hfov_deg=90,
        vfov_deg=56,
        frustum_depth=15,
        interval_ms=100,
        trail=20,
        show_frustum=True,
        ortho=True,
    )
