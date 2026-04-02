import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap_name="jet", minval=0.15, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc_{cmap_name}",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def disturbance_gain(dist_vec, drone_vel_dir, animal_vel_dir, config):
    dx, dy, dz = dist_vec
    radial_distance = np.sqrt(dx * dx + dy * dy)
    z_abs = abs(dz)

    s0 = (
        radial_distance / max(config.XY_scale, 1e-9) +
        z_abs / max(config.Z_scale, 1e-9)
    )

    # Geometry should matter mainly nearby and fade with distance
    gate = np.exp(-s0)

    angle_bad = angle_gain(dist_vec)
    heading_bad = heading_gain(dist_vec, drone_vel_dir)
    axis_bad = animal_axis_gain(dist_vec, animal_vel_dir)

    angle_eff = angle_bad * gate
    heading_eff = heading_bad * gate
    axis_eff = axis_bad * gate

    # Axis pushes preferred stand-off outward
    s = (
        (radial_distance / max(config.XY_scale, 1e-9)) * (1.0 + config.w_axis * axis_eff) +
        z_abs / max(config.Z_scale, 1e-9)
    )
    utility_base = 1.0 - np.exp(-s)

    g_an = np.clip(1.0 - config.w_angle * angle_eff, 0.0, 1.0)
    g_he = np.clip(1.0 - config.w_heading * heading_eff, 0.0, 1.0)
    g_ax = np.clip(1.0 - config.w_axis * axis_eff, 0.0, 1.0)

    utility = utility_base * g_an * g_he * g_ax
    disturbance = 1.0 - np.clip(utility, 0.0, 1.0)
    return float(disturbance)

# Analytic base-only maximum (radial + z only)
def analytic_upper_bound(XY_scale, Z_scale, r_scale, D_R_scale, eps=1e-12):
    """
    Base-only reward:
        R_base(x,z) = (1-D_R_scale) * (-sqrt(x^2+z^2) / r_scale) + D_R_scale * (1 - exp(-(x/XY_scale + z/Z_scale)))

    over x,z >= 0.

    Returns:
        base_max, x_star, z_star
    """
    if D_R_scale <= eps:
        return 0.0, 0.0, 0.0

    if (1.0 - D_R_scale) <= eps:
        # Supremum is 1 at infinity
        return 1.0, np.inf, np.inf

    S = np.sqrt(XY_scale * XY_scale + Z_scale * Z_scale)

    log_arg = D_R_scale * r_scale * S / max((1.0 - D_R_scale) * XY_scale * Z_scale, eps)

    if log_arg <= 1.0:
        return 0.0, 0.0, 0.0

    lam = (XY_scale * Z_scale / (XY_scale * XY_scale + Z_scale * Z_scale)) * np.log(log_arg)

    x_star = lam * Z_scale
    z_star = lam * XY_scale

    d = lam * S
    expo = lam * (XY_scale * XY_scale + Z_scale * Z_scale) / (XY_scale * Z_scale)

    base_max = (1.0 - D_R_scale) * (-d / r_scale) + D_R_scale * (1.0 - np.exp(-expo))

    return float(base_max), float(x_star), float(z_star)

def sigmoid_disturbance(x, midpoint, sharpness):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp((x - midpoint) / sharpness))

def altitude_gain_sigmoid(dist_vec):
    _, _, dz = dist_vec
    altitude = abs(dz)
    return sigmoid_disturbance(altitude, midpoint=60.0, sharpness=7.0)

def horizontal_gain_sigmoid(dist_vec):
    dx, dy, _ = dist_vec
    d = np.sqrt(dx*dx + dy*dy)
    return sigmoid_disturbance(d, midpoint=100.0, sharpness=15.0)

def altitude_gain(dist_vec):
    _, _, z = dist_vec
    z = abs(z)

    if z <= 40:
        return 1.0 - 0.6 * (z - 20) / 20
    elif z <= 110:
        return 0.4 * (1 - (z - 40) / 70)
    return 0.0


def horizontal_gain(dist_vec):
    dx, dy, _ = dist_vec
    d = np.sqrt(dx * dx + dy * dy)

    if d <= 135:
        return 1.0 - 0.2 * (d - 50) / 85
    elif d <= 300:
        return 0.8 * (1 - (d - 135) / 165)
    return 0.0

def heading_gain(dist_vec, drone_vel_dir):
    v = np.asarray(drone_vel_dir, dtype=float)
    d = np.asarray(dist_vec, dtype=float)

    v_norm = np.linalg.norm(v)
    d_norm = np.linalg.norm(d)

    if v_norm < 1e-8 or d_norm < 1e-8:
        return 0.0

    cos_theta = np.dot(v, d) / (v_norm * d_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return max(0.0, -cos_theta)

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

def angle_distance_plot():
    x = np.linspace(-350, 350, 400)
    z = np.linspace(-50, 100, 400)
    X, Z = np.meshgrid(x, z)

    G = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist_vec = (X[i, j], 0.0, Z[i, j])

            g_h = horizontal_gain(dist_vec)
            g_v = altitude_gain(dist_vec)
            g_a = angle_gain(dist_vec)

            comps = [g_a]
            base = g_h * g_v
            G[i, j] = (base + sum(comps)) / (len(comps) + 1)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        G,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        cmap=truncate_colormap(cmap_name="hot", minval=0, maxval=0.42),
        aspect="auto"
    )

    plt.colorbar(im, label="Angle + Distance Disturbance Gain")
    plt.xlabel("Horizontal Distance")
    plt.ylabel("Vertical Distance")
    plt.title("Combined Angle-Shaped Distance Gain")
    plt.grid(alpha=0.2)
    plt.savefig("./figures/angle_distance.png", dpi=300)
    plt.show()

def component_plots():
    # top-row grids
    x_top = np.linspace(-350, 350, 400)
    z_top = np.linspace(-120, 120, 400)
    X_top, Z_top = np.meshgrid(x_top, z_top)

    G_h = np.zeros_like(X_top, dtype=float)
    G_v = np.zeros_like(X_top, dtype=float)

    for i in range(X_top.shape[0]):
        for j in range(X_top.shape[1]):
            dist_vec = (X_top[i, j], 0.0, Z_top[i, j])
            G_h[i, j] = horizontal_gain(dist_vec)
            G_v[i, j] = altitude_gain(dist_vec)

    # bottom-row grid: exactly your angle_distance_plot setup
    x_bot = np.linspace(-350, 350, 400)
    z_bot = np.linspace(-50, 150, 400)
    X_bot, Z_bot = np.meshgrid(x_bot, z_bot)

    G_ad = np.zeros_like(X_bot, dtype=float)

    for i in range(X_bot.shape[0]):
        for j in range(X_bot.shape[1]):
            dist_vec = (X_bot[i, j], 0.0, Z_bot[i, j])

            g_h = horizontal_gain(dist_vec)
            g_v = altitude_gain(dist_vec)
            g_a = angle_gain(dist_vec)

            comps = [g_a]
            base = g_h * g_v
            G_ad[i, j] = (base + sum(comps)) / (len(comps) + 1)

    # angle data for polar plot
    angles = np.linspace(0, 360, 720)
    gains = []
    for a in angles:
        rad = np.radians(a)
        dist_vec = (np.cos(rad), 0.0, np.sin(rad))
        gains.append(angle_gain(dist_vec))

    gains = np.array(gains)
    angles_rad = np.radians(angles)

    r = np.linspace(0, 1, 2)
    theta, R = np.meshgrid(angles_rad, r)
    Z_angle = np.tile(gains, (2, 1))

    cmap = truncate_colormap(cmap_name="jet", minval=0.0, maxval=1.0)

    # 2 rows x 4 cols, last col reserved for one tall colorbar
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 0.05],
        height_ratios=[1, 1],
        wspace=0.25,
        hspace=0.3
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], projection="polar")
    ax4 = fig.add_subplot(gs[1, 0:3])   # span row 2, cols 1..3
    cax = fig.add_subplot(gs[:, 3])     # colorbar spanning both rows

    im1 = ax1.imshow(
        G_h,
        extent=[x_top.min(), x_top.max(), z_top.min(), z_top.max()],
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto"
    )
    ax1.set_title("Horizontal Gain", size=18)
    ax1.set_xlabel("Horizontal Distance", size=16)
    ax1.set_ylabel("Vertical Distance", size=16)
    ax1.grid(alpha=0.2)

    im2 = ax2.imshow(
        G_v,
        extent=[x_top.min(), x_top.max(), z_top.min(), z_top.max()],
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto"
    )
    ax2.set_title("Altitude Gain", size=18)
    ax2.set_xlabel("Horizontal Distance", size=16)
    ax2.grid(alpha=0.2)

    im3 = ax3.pcolormesh(
        theta,
        R,
        Z_angle,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=1
    )
    ax3.set_title("Angle Gain", size=18)
    ax3.set_yticks([])
    ax3.set_xticks([0, np.pi/4, (np.pi/4)+1*(np.pi/2), (np.pi/4)+2*(np.pi/2), (np.pi/4)+3*(np.pi/2), np.pi, np.pi*3/2])

    im4 = ax4.imshow(
        G_ad,
        extent=[x_bot.min(), x_bot.max(), z_bot.min(), z_bot.max()],
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto"
    )
    
    ax4.set_title("Combined Angle-Shaped Distance Gain", size=18)
    ax4.set_xlabel("Horizontal Distance", size=16)
    ax4.set_ylabel("Vertical Distance", size=16)
    ax4.grid(alpha=0.2)

    cbar = fig.colorbar(im4, cax=cax)
    cbar.set_label("Gain")

    plt.savefig("./figures/comps.png", dpi=300, bbox_inches="tight")
    plt.show()

def sigmoid_disturbance(x, midpoint, sharpness):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp((x - midpoint) / sharpness))


def altitude_disturbance_sigmoid(altitude_m):
    return sigmoid_disturbance(altitude_m, midpoint=60.0, sharpness=7.0)


def horizontal_disturbance_sigmoid(distance_m):
    return sigmoid_disturbance(distance_m, midpoint=100.0, sharpness=15.0)


def find_x_at_level(x, y, level):
    return np.interp(level, y[::-1], x[::-1])


def hor_ver_plot():

    altitudes = np.linspace(0, 200, 2000)
    distances = np.linspace(0, 500, 2000)

    alt_vals = altitude_disturbance_sigmoid(altitudes)
    dist_vals = horizontal_disturbance_sigmoid(distances)

    levels = [0.95, 0.90, 0.50, 0.10, 0.05]

    fig = plt.figure(figsize=(10,4))

    # 2 cm offset in inches
    offset_inches = 2 / 2.54


    # --- altitude subplot ---
    ax1 = plt.subplot(1,2,1)

    for lvl in levels:

        x_cross = find_x_at_level(altitudes, alt_vals, lvl)

        ax1.axhline(
            lvl,
            linestyle="--",
            linewidth=1,
            color="0.5",
            zorder=0
        )

        text_transform = transforms.offset_copy(
            ax1.transData,
            fig=fig,
            x=offset_inches,
            units='inches'
        )

        ax1.text(
            x_cross,
            lvl,
            f"{lvl:.2f}",
            fontsize=9,
            color="0.5",
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15),
            transform=text_transform,
            zorder=3
        )

    ax1.plot(altitudes, alt_vals, linewidth=1.8, zorder=2, color='orange')

    ax1.set_title("Altitude disturbance", fontsize=18)
    ax1.set_xlabel("Altitude (m)", fontsize=16)
    ax1.set_ylabel("Disturbance gain", fontsize=16)



    # --- horizontal subplot ---
    ax2 = plt.subplot(1,2,2)

    for lvl in levels:

        x_cross = find_x_at_level(distances, dist_vals, lvl)

        ax2.axhline(
            lvl,
            linestyle="--",
            linewidth=1,
            color="0.5",
            zorder=0
        )

        text_transform = transforms.offset_copy(
            ax2.transData,
            fig=fig,
            x=offset_inches,
            units='inches'
        )

        ax2.text(
            x_cross,
            lvl,
            f"{lvl:.2f}",
            fontsize=9,
            color="0.5",
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15),
            transform=text_transform,
            zorder=3
        )

    ax2.plot(distances, dist_vals, linewidth=1.8, zorder=2, color='tab:orange')

    ax2.set_title("Horizontal disturbance", fontsize=18)
    ax2.set_xlabel("Horizontal distance (m)", fontsize=16)
    ax2.set_ylabel("Disturbance gain", fontsize=16)

    plt.tight_layout()
    plt.savefig("figures/hor_ver.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # angle_plot()
    # distance_plot()
    # angle_distance_plot()
    component_plots()
    # hor_ver_plot()
    ...