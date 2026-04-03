import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap_name="jet", minval=0.15, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc_{cmap_name}",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def combined_non_disturbance(radial, altitude):
    radial_scale = 90 / np.log(2)
    altitude_scale = 50 / np.log(2)

    radial_abs = np.abs(radial)
    altitude_abs = np.abs(altitude)

    radial_term = np.exp((10 - radial_abs) / radial_scale)
    altitude_term = np.exp((10 - altitude_abs) / altitude_scale)

    return 1 - radial_term * altitude_term


def distance_fade(dist_vec, max_range=500.0):
    dx, dy, dz = dist_vec
    d = np.sqrt(dx * dx + dy * dy + dz * dz)

    fade = 1.0 - d / max_range
    return float(np.clip(fade, 0.0, 1.0))


def static_disturbance(dist_vec, config):
    dx, dy, dz = dist_vec
    radial_distance = np.linalg.norm([dx, dy], ord=2)
    altitude_distance = abs(dz)

    # distance-based safety
    dist_nd = combined_non_disturbance(radial_distance, altitude_distance)

    # base disturbance from proximity
    base = 1.0 - dist_nd

    # plottable geometry term
    angle_term = angle_gain(dist_vec)

    geom_amp = base * (config.w_angle * angle_term)

    disturbance = base * (1.0 + geom_amp)
    return disturbance


def dynamic_disturbance(dist_vec, drone_vel_dir, animal_vel_dir, config):
    dx, dy, dz = dist_vec
    radial_distance = np.linalg.norm([dx, dy], ord=2)
    altitude_distance = abs(dz)

    dist_nd = combined_non_disturbance(radial_distance, altitude_distance)

    # dynamic/contextual terms matter mostly when near
    near_weight = 1.0 - dist_nd

    heading_term = heading_gain(dist_vec, drone_vel_dir)
    axis_term = axis_gain(dist_vec, animal_vel_dir)

    dynamic = near_weight * (
        config.w_heading * heading_term +
        config.w_axis * axis_term
    )

    return dynamic


def disturbance_gain(dist_vec, drone_vel_dir, animal_vel_dir, config):
    d_static = static_disturbance(dist_vec, config)
    d_dynamic = dynamic_disturbance(dist_vec, drone_vel_dir, animal_vel_dir, config)

    disturbance = d_static * (1.0 + d_dynamic)
    return float(np.clip(disturbance, 0.0, 1.0))


def heading_gain(dist_vec, drone_vel_dir, max_range=500.0):
    v = np.asarray(drone_vel_dir, dtype=float)
    d = np.asarray(dist_vec, dtype=float)

    v_norm = np.linalg.norm(v)
    d_norm = np.linalg.norm(d)

    if v_norm < 1e-8 or d_norm < 1e-8:
        return 0.0

    cos_theta = np.dot(v, d) / (v_norm * d_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # high only when moving toward the animal
    geom = max(0.0, -cos_theta)

    fade = distance_fade(dist_vec, max_range=max_range)
    return geom * fade


def axis_gain(dist_vec, animal_vel_dir, max_range=500.0):
    d = np.asarray(dist_vec, dtype=float)
    a = np.asarray(animal_vel_dir, dtype=float)

    # horizontal plane only
    d_xy = d[:2]
    a_xy = a[:2]

    d_norm = np.linalg.norm(d_xy)
    a_norm = np.linalg.norm(a_xy)

    if d_norm < 1e-8 or a_norm < 1e-8:
        return 0.0

    d_u = d_xy / d_norm
    a_u = a_xy / a_norm

    # front and back both high, side low
    cos_phi = np.clip(np.dot(d_u, a_u), -1.0, 1.0)
    geom = abs(cos_phi)

    fade = distance_fade(dist_vec, max_range=max_range)
    return geom * fade


def angle_gain(dist_vec, max_range=500.0):
    dx, dy, z = dist_vec
    horizontal_dist = np.sqrt(dx * dx + dy * dy)

    # angle part
    a = np.degrees(np.arctan2(z, horizontal_dist)) % 360.0

    if a < 20.0:
        t = (a + 90.0) / 110.0
        g = 0.5 * (1.0 - t)

    elif a < 60.0:
        g = 0.0

    elif a < 90.0:
        g = (a - 60.0) / 30.0

    elif a < 120.0:
        g = 1.0 - (a - 90.0) / 30.0

    elif a < 160.0:
        g = 0.0

    elif a < 270.0:
        g = 0.5 * (a - 160.0) / 110.0

    else:
        t = (a - 270.0) / 110.0
        g = 0.5 * (1.0 - t)

    fade = distance_fade(dist_vec, max_range=max_range)
    return g * fade

def plot():
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_map",
        [
            (0.0, "#D95F02"),
            (0.5, "#7570B3"),
            (1.0, "#1B9E77"),
        ],
        N=256
    )

    custom_cmap_r = custom_cmap.reversed()

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.05],
        height_ratios=[1, 1.15],
        wspace=0.30,
        hspace=0.32
    )

    ax_dist = fig.add_subplot(gs[0, 0])
    ax_ang = fig.add_subplot(gs[0, 1], projection="polar")
    ax_comb = fig.add_subplot(gs[1, 0:2])
    cax = fig.add_subplot(gs[:, 2])

    # --------------------------------------------------
    # grid
    # --------------------------------------------------
    radial =   np.linspace(-300, 300, 401)
    altitude = np.linspace(-300, 300, 401)
    R, A = np.meshgrid(radial, altitude)

    # --------------------------------------------------
    # combined radial + altitude non-disturbance
    # --------------------------------------------------
    cnd = combined_non_disturbance(R, A)

    im_dist = ax_dist.pcolormesh(
        R,
        A,
        cnd,
        shading="auto",
        cmap=custom_cmap,
        vmin=0,
        vmax=1
    )
    ax_dist.set_title("Radial + Altitude non-disturbance", fontsize=18)
    ax_dist.set_xlabel("Radial distance (m)", fontsize=14)
    ax_dist.set_ylabel("Altitude (m)", fontsize=14)

    # --------------------------------------------------
    # angle disturbance
    # --------------------------------------------------
    angles = np.linspace(0, 360, 720)
    gains = []

    for a in angles:
        rad = np.radians(a)
        dist_vec = (np.cos(rad), 0.0, np.sin(rad))
        gains.append(angle_gain(dist_vec))

    gains = np.array(gains)
    angles_rad = np.radians(angles)

    r_band = np.linspace(0, 1, 2)
    theta, R_band = np.meshgrid(angles_rad, r_band)
    Z_angle = np.tile(gains, (2, 1))

    ax_ang.pcolormesh(
        theta,
        R_band,
        Z_angle,
        shading="auto",
        cmap=custom_cmap_r,
        vmin=0,
        vmax=1
    )
    ax_ang.set_title("Angle disturbance", fontsize=18)
    ax_ang.set_yticks([])
    ax_ang.set_xticks([
        0,
        np.pi,
        np.pi / 4,
        3 * np.pi / 4,
        5 * np.pi / 4,
        3 * np.pi / 2,
        7 * np.pi / 4
    ])

    # --------------------------------------------------
    # combined
    # --------------------------------------------------
    combined = np.zeros_like(R, dtype=float)

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            dist_vec = (R[i, j], 0.0, A[i, j])
            g_ang_bad = angle_gain(dist_vec)
            combined[i, j] = cnd[i, j] * (1.0 - g_ang_bad)

    im_comb = ax_comb.pcolormesh(
        R,
        A,
        combined,
        shading="auto",
        cmap=custom_cmap,
        vmin=0,
        vmax=1
    )
    ax_comb.set_title("Combined non-disturbance", fontsize=18)
    ax_comb.set_xlabel("Radial distance (m)", fontsize=14)
    ax_comb.set_ylabel("Altitude (m)", fontsize=14)

    # --------------------------------------------------
    # shared colorbar
    # --------------------------------------------------
    cbar = fig.colorbar(im_comb, cax=cax)
    cbar.set_label("Non-disturbance", fontsize=14)

    plt.tight_layout()
    plt.savefig("./figures/comps.png", dpi=300)
    plt.show()
    
if __name__ == "__main__":
    plot()
    ...