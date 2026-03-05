import numpy as np
import matplotlib.pyplot as plt

# Spatial Gain Functions (scalar: one animal–drone pair)
def disturbance_gain(dist_vec, drone_vel_dir, drone_vel_speed, config):
    """
    Combined disturbance for one animal-drone pair.
    """
    g_altitude = np.clip(altitude_gain(dist_vec), 0.0, 1.0)
    g_horizontal = np.clip(horizontal_gain(dist_vec), 0.0, 1.0)

    g_angle = np.clip(angle_gain(dist_vec), 0.0, 1.0)
    g_heading = np.clip(heading_gain(dist_vec, drone_vel_dir), 0.0, 1.0)
    g_speed = np.clip(speed_gain(drone_vel_speed), 0.0, 1.0)

    base = g_horizontal * g_altitude

    angle_boost = g_angle * config.max_angle_boost   # [0,1]
    heading_boost = g_heading * config.max_heading_boost   # [0,1]
    speed_boost = g_speed * config.max_speed_boost   # [0,1]

    D = base + (1.0 - base) * angle_boost * base
    D = D + (1.0 - D) * heading_boost * base
    D = D + (1.0 - D) * speed_boost * base

    return D

def altitude_gain(dist_vec):
    _, _, z = dist_vec
    z = abs(z)

    if z <= 20:
        return 1.0
    elif z <= 40:
        return 1.0 - 0.6 * (z - 20) / 20
    elif z <= 110:
        return 0.4 * (1 - (z - 40) / 70)
    return 0.0


def horizontal_gain(dist_vec):
    dx, dy, _ = dist_vec
    d = np.sqrt(dx*dx + dy*dy)

    if d <= 20:
        return 1.0
    elif d <= 50:
        return 1.0 - 0.2 * (d - 20) / 30
    elif d <= 110:
        return 0.8 * (1 - (d - 50) / 60)
    return 0.0

def angle_gain(dist_vec):
    dx, dy, z = dist_vec
    horizontal_dist = np.sqrt(dx * dx + dy * dy)

    theta = np.degrees(np.arctan2(z, abs(horizontal_dist)))

    if theta <= 20:
        return 0.5 - 0.5 * (theta + 90) / 110
    elif theta <= 60:
        return 0.0
    elif theta <= 90:
        return (theta - 60) / 30
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

    return max(0.0, cos_theta)

def speed_gain(drone_vel_speed, v_min=2, v_max=8):
    if drone_vel_speed <= v_min:
        g = 0
    else:
        g = min(drone_vel_speed - v_min / (v_max - v_min), 1)
    return g

# Visualization helpers (grid evaluation)
def evaluate_on_grid(func, X, Y):
    """Evaluate scalar gain function over a meshgrid."""
    Z = np.zeros_like(X, dtype=float)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist_vec = (X[i, j], 0.0, Y[i, j])
            Z[i, j] = func(dist_vec)

    return Z


# Main Visualization
if __name__ == "__main__":

    # Spatial grid
    X, Y = np.meshgrid(
        np.linspace(-110, 110, 300),
        np.linspace(0, 110, 300)
    )

    # Evaluate components
    ALT = evaluate_on_grid(altitude_gain, X, Y)
    HOR = evaluate_on_grid(horizontal_gain, X, Y)
    ANG = evaluate_on_grid(angle_gain, X, Y)

    base = ALT * HOR
    max_angle_boost = 1

    D = base + (1.0 - base) * (ANG * max_angle_boost) * base

    COMBINED = D

    # FIGURE 1 — Combined disturbance
    fig1, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(X, Y, COMBINED, cmap="magma", shading="auto")
    ax.set_title("Combined 360° Disturbance Field")
    ax.set_xlabel("Horizontal Offset (m)")
    ax.set_ylabel("Altitude (m)")
    ax.plot(0, 0, "wo", markersize=8)

    fig1.colorbar(im, ax=ax, label="Disturbance Intensity")
    plt.tight_layout()
    plt.savefig("figures/disturbance_combined.png", dpi=300)


    # FIGURE 2 — Individual components
    fig2, axs = plt.subplots(1, 3, figsize=(18, 5))

    components = [
        (ALT, "Altitude Component", "Blues"),
        (HOR, "Horizontal Component", "Greens"),
        (ANG, "Angle Component", "Reds"),
    ]

    for ax, (Z, title, cmap) in zip(axs, components):
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Horizontal Offset (m)")
        ax.set_ylabel("Altitude (m)")
        ax.plot(0, 0, "wo", markersize=6)
        fig2.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("figures/disturbance_components.png", dpi=300)

    plt.show()
