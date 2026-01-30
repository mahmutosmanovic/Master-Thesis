from settings import *

def draw_all_static(df, k=1):
    df_filtered = df[df.index % k == 0]
    plt.scatter(df_filtered['x'] / 10**6, df_filtered['y'] / 10**6, 
                c=df_filtered.index, s=3, alpha=0.7, cmap='seismic_r')
    plt.colorbar(label='Time')
    plt.title('Coordinate System: EPSG32636')
    plt.show()

def draw_trail_2D(df, k=30, interval=50, cmap='seismic'):
    df = df.copy()
    # Pre-scale data to avoid doing math inside the update loop
    df['x_s'] = df['x'] / 1e6
    df['y_s'] = df['y'] / 1e6
    
    # Pre-calculate coordinates to avoid iloc/indexing overhead in the loop
    coords = df[['x_s', 'y_s']].to_numpy()

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
    ax.set_xlim(df['x_s'].min(), df['x_s'].max())
    ax.set_ylim(df['y_s'].min(), df['y_s'].max())

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
    return ani

def draw_trail_3D(df, interval=50, cmap='seismic'):
    df = df.copy()

    # Scale coordinates
    df['x_s'] = df['x'] / 1e6
    df['y_s'] = df['y'] / 1e6

    if 'z' in df.columns:
        df['z_s'] = df['z'] / 1e3
    else:
        df['z_s'] = 0.0

    coords = df[['x_s', 'y_s', 'z_s']].to_numpy()
    n_points = len(coords)

    # ---- Figure ----
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=25, azim=-45)

    ax.set_title('Cumulative 3D Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Limits
    ax.set_xlim(df['x_s'].min(), df['x_s'].max())
    ax.set_ylim(df['y_s'].min(), df['y_s'].max())

    if df['z_s'].nunique() <= 1:
        ax.set_zlim(-0.1, 0.1)
    else:
        ax.set_zlim(df['z_s'].min(), df['z_s'].max())

    # ---- Line collection ----
    dummy = np.zeros((1, 2, 3))

    norm = colors.Normalize(vmin=0, vmax=1)

    lc = Line3DCollection(
        dummy,
        cmap=cmap,
        norm=norm,
        linewidth=2
    )

    ax.add_collection3d(lc)

    # ---- Update ----
    def update(frame):

        if frame < 1:
            lc.set_segments([])
            return lc,

        pts = coords[:frame + 1]

        pts = pts.reshape(-1, 1, 3)

        segments = np.concatenate(
            [pts[:-1], pts[1:]],
            axis=1
        )

        lc.set_segments(segments)

        # -------- Age colors --------

        n_seg = len(segments)

        # newest = 0, oldest = 1
        ages = np.arange(n_seg)[::-1]

        if n_seg > 1:
            ages = ages / (n_seg - 1)
        else:
            ages = np.array([0.0])

        # IMPORTANT: update array + clim
        lc.set_array(ages)
        lc.set_clim(0, 1)

        return lc,

    # ---- Animate ----
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_points,
        interval=interval,
        blit=False,
        repeat=False
    )

    plt.show()

    return ani