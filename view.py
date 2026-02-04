from settings import *

def draw_all_static(df, k=1):
    df_filtered = df[df.index % k == 0]
    plt.scatter(df_filtered['x'], df_filtered['y'], 
                c=df_filtered.index, s=3, alpha=0.7, cmap='seismic_r')
    plt.colorbar(label='Time')
    plt.title('Coordinate System: EPSG32636')
    plt.show()

def draw_trail_2D(df, k=30, interval=50, cmap='seismic'):
    df = df.copy()
    
    # Pre-calculate coordinates to avoid iloc/indexing overhead in the loop
    coords = df[['x', 'y']].to_numpy()

    fig, ax = plt.subplots()
    
    # Initialize scatter with dummy data that matches the trail size 'k'
    # This ensures the internal color buffers are properly sized.
    scatter = ax.scatter(
        coords[:1, 0], coords[:1, 1], 
        c=[1.0], 
        s=20, 
        cmap=cmap, 
        vmin=0, vmax=1, # Explicitly lock the color scale
        alpha=0.9
    )

    ax.set_title('Coordinate System: EPSG32636')
    ax.set_xlabel('X (scaled)')
    ax.set_ylabel('Y (scaled)')
    ax.set_xlim(df['x'].min(), df['x'].max())
    ax.set_ylim(df['y'].min(), df['y'].max())

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap), ax=ax)
    cbar.set_label('Trail age (0 = oldest, 1 = newest)')

    # Pre-calculate age array for the maximum trail size
    full_age = np.linspace(0, 1, k)

    def update(frame):
        start = max(0, frame - k + 1)
        # Slicing numpy arrays is much faster than df.iloc
        trail_data = coords[start:frame + 1]
        n = len(trail_data)

        # Update positions
        scatter.set_offsets(trail_data)

        # Update colors: Slice the pre-calculated age array to match current trail length
        # We take the LAST n elements to ensure the leading point is always 1.0
        scatter.set_array(full_age[-n:])

        return (scatter,)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(df),
        interval=interval,
        blit=True,
        repeat=False
    )

    plt.show()

def draw_trail_3D(df, interval=50, trail_length=50):

    df = df.copy()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib import animation
    import numpy as np

    # ---------- Setup ----------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.view_init(elev=25, azim=-45)

    ax.set_title("Cumulative 3D Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # ---------- Axis limits ----------
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    z_min, z_max = df["z"].min(), df["z"].max()

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    )

    pad = 0.05 * max_range
    r = max_range / 2 + pad

    ax.set_xlim(x_mid - r, x_mid + r)
    ax.set_ylim(y_mid - r, y_mid + r)
    ax.set_zlim(z_mid - r, z_mid + r)

    # ---------- Colors ----------
    base_colors = [
        "red", "green", "blue", "orange", "purple",
        "yellow", "cyan", "magenta", "lime", "brown"
    ]

    entities = df["entity"].unique()

    color_map = {
        ent: base_colors[i % len(base_colors)]
        for i, ent in enumerate(entities)
    }

    # ---------- Data ----------
    entity_data = {}

    for ent in entities:
        sub = df[df["entity"] == ent]
        entity_data[ent] = sub[["x", "y", "z"]].to_numpy()

    # Extract drone yaw separately
    drone_df = df[df["entity"] == "DRONE"].reset_index(drop=True)
    drone_yaws = drone_df["yaw"].to_numpy()

    # ---------- Trails ----------
    lines = {}

    for ent in entities:

        dummy = np.zeros((1, 2, 3))

        lc = Line3DCollection(
            dummy,
            color=color_map[ent],
            linewidth=2,
            alpha=0.6
        )

        ax.add_collection3d(lc)
        lines[ent] = lc

    # ---------- Base camera direction ----------
    base_camera_dir = np.array([-1, -1, -1], dtype=float)
    base_camera_dir /= np.linalg.norm(base_camera_dir)

    # ---------- Frustum ----------
    fov = np.deg2rad(FOV_DEG)
    MAX_VIEW_DIST = MAX_VIEW_RANGE
    fov_len = min(r * 0.7, MAX_VIEW_DIST)

    drone_color = color_map.get("DRONE", "cyan")
    animal_color = color_map.get("PIGEON", "cyan")

    fov_lines = Line3DCollection(
        np.zeros((8, 2, 3)),
        colors=drone_color,
        linewidth=1.5,
        linestyles="--",
        alpha=0.9
    )

    ax.add_collection3d(fov_lines)

    # ---------- Head Dots ----------
    drone_dot = ax.scatter([], [], [], s=100, c=drone_color, marker="o")
    animal_dot = ax.scatter([], [], [], s=100, c=animal_color, marker="o")

    # ---------- Update ----------
    def update(frame):

        drone_frame = min(frame, len(drone_yaws) - 1)

        for ent in entities:

            coords = entity_data[ent]
            n = len(coords)

            if frame < 1 or frame >= n:
                lines[ent].set_segments([])
                continue

            # Trail window
            start = max(0, frame - trail_length)

            pts = coords[start:frame + 1]
            pts = pts.reshape(-1, 1, 3)

            segments = np.concatenate(
                [pts[:-1], pts[1:]],
                axis=1
            )

            # Alpha fade
            n_seg = len(segments)

            if n_seg > 1:
                alphas = np.linspace(0.1, 0.8, n_seg)
            else:
                alphas = [0.8]

            colors = []

            base = color_map[ent]

            for a in alphas:
                colors.append(
                    plt.matplotlib.colors.to_rgba(base, a)
                )

            lines[ent].set_segments(segments)
            lines[ent].set_color(colors)

            # ---------- Head dots ----------
            if ent == "DRONE":
                drone_dot._offsets3d = (
                    [coords[frame][0]],
                    [coords[frame][1]],
                    [coords[frame][2]],
                )

            if ent == "PIGEON":
                animal_dot._offsets3d = (
                    [coords[frame][0]],
                    [coords[frame][1]],
                    [coords[frame][2]],
                )

            # ---------- Draw Frustum ----------
            if ent == "DRONE":

                pos = coords[frame]

                # Get true yaw
                yaw = drone_yaws[drone_frame]

                # Rotate base camera dir
                c = np.cos(yaw)
                s = np.sin(yaw)

                Rz = np.array([
                    [ c, -s, 0],
                    [ s,  c, 0],
                    [ 0,  0, 1]
                ])

                camera_dir = Rz @ base_camera_dir
                camera_dir /= np.linalg.norm(camera_dir)

                # Build orthonormal basis
                if abs(camera_dir[2]) < 0.9:
                    tmp = np.array([0, 0, 1])
                else:
                    tmp = np.array([0, 1, 0])

                right = np.cross(camera_dir, tmp)
                right /= np.linalg.norm(right)

                up = np.cross(right, camera_dir)
                up /= np.linalg.norm(up)

                # Frustum size
                half = np.tan(fov / 2) * fov_len

                center = pos + camera_dir * fov_len

                p1 = center + half * right + half * up
                p2 = center - half * right + half * up
                p3 = center - half * right - half * up
                p4 = center + half * right - half * up

                segs = [
                    [pos, p1],
                    [pos, p2],
                    [pos, p3],
                    [pos, p4],

                    [p1, p2],
                    [p2, p3],
                    [p3, p4],
                    [p4, p1],
                ]

                fov_lines.set_segments(segs)

        return list(lines.values()) + [fov_lines, drone_dot, animal_dot]

    # ---------- Animate ----------
    max_frames = max(len(v) for v in entity_data.values())

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=interval,
        blit=False,
        repeat=False
    )

    # ---------- Save video ----------
    from matplotlib.animation import FFMpegWriter
    fps = max(1, int(1000 / interval))
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    ani.save("video.mp4", writer=writer)

    plt.show()
