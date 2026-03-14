import numpy as np
import matplotlib.pyplot as plt

# Spatial Gain Functions (scalar: one animal–drone pair)
def disturbance_gain(dist_vec, drone_vel_dir, drone_vel_speed, animal_vel_dir, config):
    """
    Combined disturbance for one animal-drone pair.
    """

    g_altitude = np.clip(altitude_gain(dist_vec), 0.0, 1.0)
    g_horizontal = np.clip(horizontal_gain(dist_vec), 0.0, 1.0)

    g_angle = np.clip(angle_gain(dist_vec), 0.0, 1.0)
    g_heading = np.clip(heading_gain(dist_vec, drone_vel_dir), 0.0, 1.0)
    g_speed = np.clip(speed_gain(drone_vel_speed), 0.0, 1.0)
    g_axis = np.clip(animal_axis_gain(dist_vec, animal_vel_dir), 0.0, 1.0)

    base = g_horizontal * g_altitude

    angle_boost = g_angle * config.max_angle_boost
    heading_boost = g_heading * config.max_heading_boost
    speed_boost = g_speed * config.max_speed_boost
    axis_boost = g_axis * config.max_axis_boost

    multiplier = 1 + angle_boost + heading_boost + speed_boost + axis_boost

    D = base * multiplier

    return np.clip(D, 0.0, 1.0)

def disturbance_gain_alt(dist_vec, field_boost=0.0):
    g_h = horizontal_gain(dist_vec, field_boost=field_boost)
    g_v = altitude_gain(dist_vec, field_boost=field_boost)
    g_a = angle_gain(dist_vec)

    base = g_h * g_v
    comps = [g_a]

    D = (base + sum(comps)) / (len(comps) + 1)
    return min(D, 1)

def piecewise_gain(x, points, field_boost=0.0):
    s = 1.0 + field_boost
    pts = [(px * s, py) for px, py in points]

    if x <= pts[0][0]:
        return pts[0][1]

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return pts[-1][1]

def altitude_gain(dist_vec, field_boost=0.0):
    _, _, z = dist_vec
    z = abs(z)
    return piecewise_gain(
        z,
        [(20, 1.0), (40, 0.4), (110, 0.1), (500, 0.0)],
        field_boost=field_boost
    )

def horizontal_gain(dist_vec, field_boost=0.0):
    dx, dy, _ = dist_vec
    d = np.sqrt(dx * dx + dy * dy)
    return piecewise_gain(
        d,
        [(20, 1.0), (50, 0.8), (110, 0.1), (500, 0.0)],
        field_boost=field_boost
    )

def plot_gains():

    # sample distances
    distances = np.linspace(0, 550, 600)

    altitude_vals = []
    horizontal_vals = []

    for d in distances:
        altitude_vals.append(altitude_gain((0, 0, d)))
        horizontal_vals.append(horizontal_gain((d, 0, 0)))

    altitude_vals = np.array(altitude_vals)
    horizontal_vals = np.array(horizontal_vals)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # altitude plot
    axes[0].plot(distances, altitude_vals)
    axes[0].set_title("Altitude Gain")
    axes[0].set_ylabel("Gain")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True)

    # horizontal plot
    axes[1].plot(distances, horizontal_vals)
    axes[1].set_title("Horizontal Gain")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Gain")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def heading_gain(dist_vec, drone_vel_dir):
    v = np.asarray(drone_vel_dir, dtype=float)
    d = np.asarray(dist_vec, dtype=float)

    v_norm = np.linalg.norm(v)
    d_norm = np.linalg.norm(d)

    if v_norm < 1e-8 or d_norm < 1e-8:
        return 0.0

    cos_theta = np.dot(v, d) / (v_norm * d_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return max(0.0, cos_theta)

def animal_axis_gain(dist_vec, animal_vel_dir):
    d = dist_vec
    a = animal_vel_dir

    # horizontal plane only
    d_xy = d[:2]
    a_xy = a[:2]

    d_norm = np.linalg.norm(d_xy)
    a_norm = np.linalg.norm(a_xy)

    if d_norm < 1e-8 or a_norm < 1e-8:
        return 0.0

    d_u = d_xy / d_norm
    a_u = a_xy / a_norm

    # abs -> front and back both high, side low
    cos_phi = np.clip(np.dot(d_u, a_u), -1.0, 1.0)
    return abs(cos_phi)

def speed_gain(drone_vel_speed, v_min=2, v_max=8):
    if drone_vel_speed <= v_min:
        return 0.0

    g = (drone_vel_speed - v_min) / (v_max - v_min)
    return min(max(g, 0.0), 1.0)

# Visualization helpers (grid evaluation)
def evaluate_on_grid(func, X, Y):
    """Evaluate scalar gain function over a meshgrid."""
    Z = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist_vec = (X[i, j], 0.0, Y[i, j])
            Z[i, j] = func(dist_vec)

    return Z

def angle_gain(dist_vec):
    dx, dy, z = dist_vec
    horizontal_dist = np.sqrt(dx * dx + dy * dy)

    a = np.degrees(np.arctan2(z, horizontal_dist)) % 360.0

    if a < 20.0:
        # wrapped segment: 270 -> 360 -> 20
        # 270 : 0.5, 20 : 0
        t = (a + 90.0) / 110.0
        return 0.5 * (1.0 - t)

    elif a < 60.0:
        # 20 -> 60 : 0
        return 0.0

    elif a < 90.0:
        # 60 -> 90 : 0 -> 1
        return (a - 60.0) / 30.0

    elif a < 120.0:
        # 90 -> 120 : 1 -> 0
        return 1.0 - (a - 90.0) / 30.0

    elif a < 160.0:
        # 120 -> 160 : 0
        return 0.0

    elif a < 270.0:
        # 160 -> 270 : 0 -> 0.5
        return 0.5 * (a - 160.0) / 110.0

    else:
        # 270 -> 360 : 0.5 -> interpolated value toward 20
        t = (a - 270.0) / 110.0
        return 0.5 * (1.0 - t)
    
def angle_plot():
    angles = np.linspace(0, 360, 720)
    gains = []

    for a in angles:
        rad = np.radians(a)

        # create synthetic vector that produces this angle
        dist_vec = (np.cos(rad), 0.0, np.sin(rad))
        gains.append(angle_gain(dist_vec))

    gains = np.array(gains)
    angles_rad = np.radians(angles)

    r = np.linspace(0, 1, 2)
    theta, R = np.meshgrid(angles_rad, r)
    Z = np.tile(gains, (2,1))

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8,8))

    c = ax.pcolormesh(theta, R, Z, shading="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_yticks([])
    ax.set_title("Angle Gain Disturbance (360°)")

    plt.colorbar(c, label="Disturbance Gain")
    plt.savefig("./figures/angle.png", dpi=300)
    plt.show()

def distance_plot():
    x = np.linspace(-120, 120, 400)   # horizontal distance
    z = np.linspace(-120, 120, 400)   # vertical distance
    X, Z = np.meshgrid(x, z)

    G = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist_vec = (X[i, j], 0.0, Z[i, j])
            g_h = horizontal_gain(dist_vec)
            g_v = altitude_gain(dist_vec)
            G[i, j] = g_h * g_v

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        G,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    plt.colorbar(im, label="Distance Disturbance Gain")
    plt.xlabel("Horizontal Distance")
    plt.ylabel("Vertical Distance")
    plt.title("Horizontal + Vertical Distance Gain")
    plt.grid(alpha=0.2)
    plt.savefig("./figures/distance.png", dpi=300)
    plt.show()    

def angle_distance_plot(field_boost=0):
    x = np.linspace(-160, 160, 400)
    z = np.linspace(-160, 160, 400)
    X, Z = np.meshgrid(x, z)

    G = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist_vec = (X[i, j], 0.0, Z[i, j])

            g_h = horizontal_gain(dist_vec, field_boost)
            g_v = altitude_gain(dist_vec, field_boost)
            g_a = angle_gain(dist_vec)

            comps = [g_a]
            base = g_h * g_v
            D = (base + sum(comps)) / (len(comps) + 1)
            G[i, j] = min(D, 1)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        G,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        cmap="RdYlGn_r",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(im, label="Angle + Distance Disturbance Gain")
    plt.xlabel("Horizontal Distance")
    plt.ylabel("Vertical Distance")
    plt.title("Combined Angle-Shaped Distance Gain")
    plt.grid(alpha=0.2)
    plt.savefig("./figures/angle_distance.png", dpi=300)
    plt.show()

def component_plots():
    x = np.linspace(-120, 120, 400)
    z = np.linspace(-120, 120, 400)
    X, Z = np.meshgrid(x, z)

    G_h = np.zeros_like(X, dtype=float)
    G_v = np.zeros_like(X, dtype=float)
    G_a = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist_vec = (X[i, j], 0.0, Z[i, j])

            G_h[i, j] = horizontal_gain(dist_vec)
            G_v[i, j] = altitude_gain(dist_vec)
            G_a[i, j] = angle_gain(dist_vec)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    titles = [
        "Horizontal Gain",
        "Altitude Gain",
        "Angle Gain"
    ]
    grids = [G_h, G_v, G_a]

    for ax, G, title in zip(axes, grids, titles):
        im = ax.imshow(
            G,
            extent=[x.min(), x.max(), z.min(), z.max()],
            origin="lower",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            aspect="auto"
        )
        ax.set_title(title)
        ax.set_xlabel("Horizontal Distance")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Vertical Distance")

    cbar = fig.colorbar(im)
    cbar.set_label("Gain")

    plt.tight_layout()
    plt.savefig("./figures/comps.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # angle_plot()
    # distance_plot()
    angle_distance_plot(field_boost=10)
    # component_plots()
    # plot_gains()
    ...