import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def set_axes_equal(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

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

def draw_frustum(ax, pos, view_dir, hfov, vfov, depth, color="blue", alpha=0.25):
    view_dir = np.asarray(view_dir, dtype=float)
    n = np.linalg.norm(view_dir)
    if n < 1e-8:
        return
    forward = view_dir / n

    world_up = np.array([0.0, 0.0, 1.0])

    # Right in XY plane
    right = np.cross(world_up, forward)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn

    # ✅ FIX: normalize up
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    tan_h = np.tan(hfov * 0.5)
    tan_v = np.tan(vfov * 0.5)

    center = pos + forward * depth

    corners = [
        center + right * (depth * tan_h) + up * (depth * tan_v),
        center - right * (depth * tan_h) + up * (depth * tan_v),
        center - right * (depth * tan_h) - up * (depth * tan_v),
        center + right * (depth * tan_h) - up * (depth * tan_v),
    ]

    # Rays
    for c in corners:
        ax.plot(
            [pos[0], c[0]],
            [pos[1], c[1]],
            [pos[2], c[2]],
            color=color,
            alpha=alpha,
        )

    # Far plane
    xs = [c[0] for c in corners] + [corners[0][0]]
    ys = [c[1] for c in corners] + [corners[0][1]]
    zs = [c[2] for c in corners] + [corners[0][2]]

    ax.plot(xs, ys, zs, color=color, alpha=alpha)

    ax.set_proj_type('ortho')



def render_simulation_csv_3d(
    csv_path,
    hfov_deg=90.0,
    vfov_deg=60.0,
    frustum_depth=15.0,
    show=True,
):
    """
    Renders agent trajectories from CSV and draws camera frustums for drones.
    """

    # Load CSV
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by agent_id
    agents = defaultdict(list)
    for r in rows:
        agent_id = int(r["agent_id"])
        agents[agent_id].append(r)

    species_colors = {
        "jackal": "orange",
        "pigeon": "green",
        "eagle": "red",
        "drone": "blue",
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    for agent_id, records in agents.items():
        records.sort(key=lambda r: float(r["t"]))

        x = np.array([float(r["x"]) for r in records])
        y = np.array([float(r["y"]) for r in records])
        z = np.array([float(r["z"]) for r in records])

        a_type = records[0]["type"]
        color = species_colors.get(a_type, "black")

        lw = 2.5 if a_type == "drone" else 1.2
        alpha = 1.0 if a_type == "drone" else 0.7

        ax.plot(x, y, z, color=color, linewidth=lw, alpha=alpha, label=a_type)

        # Start marker
        ax.scatter(x[0], y[0], z[0], color=color, s=30, marker="o")

        # Draw frustum for drones (latest frame)
        if a_type == "drone":
            vx = float(records[-1]["view_x"]) if records[-1]["view_x"] != "" else np.nan
            vy = float(records[-1]["view_y"]) if records[-1]["view_y"] != "" else np.nan
            vz = float(records[-1]["view_z"]) if records[-1]["view_z"] != "" else np.nan

            if not np.isnan(vx):
                pos = np.array([x[-1], y[-1], z[-1]])
                view_dir = np.array([vx, vy, vz])

                draw_frustum(
                    ax,
                    pos=pos,
                    view_dir=view_dir,
                    hfov=hfov,
                    vfov=vfov,
                    depth=frustum_depth,
                    color=color,
                )

    # Labels and layout
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Agent Trajectories with Camera Frustums")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.set_box_aspect((1, 1, 1))

    set_axes_equal(ax)

    if show:
        plt.show()

    return fig, ax

if __name__ == '__main__':
    render_simulation_csv_3d("logs/simulations/ppo_rollout.csv")