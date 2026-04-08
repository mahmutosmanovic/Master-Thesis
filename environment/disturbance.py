import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# Core math
# ============================================================

def sigmoid(x, midpoint, sharpness, decreasing=True):
    x = np.asarray(x, dtype=float)
    sharpness = max(float(sharpness), 1e-8)

    if decreasing:
        return 1.0 / (1.0 + np.exp((x - midpoint) / sharpness))
    return 1.0 / (1.0 + np.exp(-(x - midpoint) / sharpness))


def total_distance(x, z):
    return np.sqrt(x**2 + z**2)


# ============================================================
# Monitoring reward
# ============================================================

def monitor_reward(x, z, preferred_distance=42.0, distance_sigma=80.0):
    d = total_distance(x, z)
    return np.exp(-0.5 * ((d - preferred_distance) / distance_sigma) ** 2)


# ============================================================
# Disturbance parts
# ============================================================

def radial_disturbance(x, radial_midpoint=85.0, radial_sharpness=60.0):
    r = np.abs(x)
    return sigmoid(r, midpoint=radial_midpoint, sharpness=radial_sharpness, decreasing=True)


def altitude_disturbance(z, altitude_midpoint=55.0, altitude_sharpness=40.0):
    return sigmoid(np.abs(z), midpoint=altitude_midpoint, sharpness=altitude_sharpness, decreasing=True)


def folded_abs_angle_deg(x, z):
    theta = np.degrees(np.arctan2(z, x))
    folded = theta.copy()
    folded = np.where(folded > 90.0, folded - 180.0, folded)
    folded = np.where(folded < -90.0, folded + 180.0, folded)
    return np.abs(folded)


def angle_badness(x, z, preferred_angle_deg=40.0, angle_scale_deg=22.0, angle_power=1.6):
    ang = folded_abs_angle_deg(x, z)
    delta = np.abs(ang - preferred_angle_deg)
    bad = 1.0 - np.exp(-0.5 * (delta / angle_scale_deg) ** 2)
    return np.clip(bad, 0.0, 1.0) ** angle_power


def raw_disturbance_calc(x, z, angle_weight=1.0):
    r_term = radial_disturbance(x)
    z_term = altitude_disturbance(z)
    a_term = angle_badness(x, z)
    return (r_term * z_term) * (1.0 + angle_weight * a_term)


# ============================================================
# Interpolated objective
# ============================================================

def interpolate_monitor_disturbance(alpha, monitor, raw_disturbance, x, z, eps=1e-6):
    alpha = float(np.clip(alpha, 0.0, 1.0))

    S = np.clip(1.0 - (raw_disturbance / 2.0), eps, 1.0)
    M = np.clip(monitor, eps, 1.0)

    combined = alpha * M + (1.0 - alpha) * S

    dist = np.sqrt(x**2 + z**2)
    return combined - (1e-5 * dist)


# ============================================================
# Env adapters
# ============================================================

def rel_vec_to_xz(rel_vec):
    """
    Convert 3D world relative vector into the 2D geometry expected here:
      x = horizontal/radial distance
      z = absolute altitude difference
    """
    rel_vec = np.asarray(rel_vec, dtype=float)
    x = np.sqrt(rel_vec[..., 0] ** 2 + rel_vec[..., 1] ** 2)
    z = np.abs(rel_vec[..., 2])
    return x, z


def disturbance_gain(rel_vec, drone_vel_dir=None, animal_vel_dir=None, config=None):
    """
    Env-compatible disturbance interface.
    """
    x, z = rel_vec_to_xz(rel_vec)
    return float(np.clip(raw_disturbance_calc(x, z), 0.0, 1.0))


def animal_axis_gain(dist_vec, animal_vel_dir):
    """
    Kept only for compatibility with older code paths.
    """
    dist_vec = np.asarray(dist_vec, dtype=float)
    animal_vel_dir = np.asarray(animal_vel_dir, dtype=float)

    rel_xy = dist_vec[:2]
    vel_xy = animal_vel_dir[:2]

    rel_norm = np.linalg.norm(rel_xy)
    vel_norm = np.linalg.norm(vel_xy)

    if rel_norm < 1e-8 or vel_norm < 1e-8:
        return 0.0

    rel_xy = rel_xy / rel_norm
    vel_xy = vel_xy / vel_norm

    return float(np.abs(np.clip(np.dot(rel_xy, vel_xy), -1.0, 1.0)))


# ============================================================
# Grid helpers for plotting
# ============================================================

def build_grid(x_min=-500.0, x_max=500.0, z_min=0.0, z_max=400.0, n_x=700, n_z=450):
    x_vals = np.linspace(x_min, x_max, n_x)
    z_vals = np.linspace(z_min, z_max, n_z)
    X, Z = np.meshgrid(x_vals, z_vals)
    return x_vals, z_vals, X, Z


def find_grid_max(X, Z, F):
    idx = np.unravel_index(np.argmax(F), F.shape)
    return X[idx], Z[idx], F[idx]


# ============================================================
# Plotting
# ============================================================

def plot_reward_landscape():
    x_vals, z_vals, X, Z = build_grid()

    radial_map = radial_disturbance(X)
    altitude_map = altitude_disturbance(Z)
    angle_map = angle_badness(X, Z)
    monitor = monitor_reward(X, Z)
    raw_dist = raw_disturbance_calc(X, Z)

    alpha0 = 0.50
    combined = interpolate_monitor_disturbance(alpha0, monitor, raw_dist, X, Z)

    x_max, z_max, f_max = find_grid_max(X, Z, combined)
    norm_max = np.sqrt(x_max**2 + z_max**2)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    plt.subplots_adjust(bottom=0.12, wspace=0.25, hspace=0.28)
    extent = [x_vals.min(), x_vals.max(), z_vals.min(), z_vals.max()]

    # TOP ROW
    im_rad = axes[0, 0].imshow(
        radial_map, extent=extent, origin="lower", aspect="auto",
        vmin=0.0, vmax=1.0, cmap="magma"
    )
    axes[0, 0].set_title("Radial disturbance")
    fig.colorbar(im_rad, ax=axes[0, 0])

    im_alt = axes[0, 1].imshow(
        altitude_map, extent=extent, origin="lower", aspect="auto",
        vmin=0.0, vmax=1.0, cmap="magma"
    )
    axes[0, 1].set_title("Altitude disturbance")
    fig.colorbar(im_alt, ax=axes[0, 1])

    im_angle = axes[0, 2].imshow(
        angle_map, extent=extent, origin="lower", aspect="auto",
        vmin=0.0, vmax=1.0, cmap="magma"
    )
    axes[0, 2].set_title("Raw angle badness")
    fig.colorbar(im_angle, ax=axes[0, 2])

    # BOTTOM ROW
    im_raw = axes[1, 0].imshow(
        raw_dist, extent=extent, origin="lower", aspect="auto", cmap="magma"
    )
    axes[1, 0].set_title("Raw disturbance (Rad*Alt*(1+Ang))")
    fig.colorbar(im_raw, ax=axes[1, 0])

    im_comb = axes[1, 1].imshow(
        combined, extent=extent, origin="lower", aspect="auto", cmap="cividis"
    )
    axes[1, 1].set_title(f"Combined objective, alpha = {alpha0:.2f}")

    max_point, = axes[1, 1].plot(
        x_max, z_max, marker="o", markersize=8, color="red", linestyle="None"
    )
    max_text = axes[1, 1].text(
        0.02, 0.98,
        f"max @ (x={x_max:.1f}, z={z_max:.1f})\n"
        f"NORM={norm_max:.1f}m\n"
        f"value={f_max:.4f}",
        transform=axes[1, 1].transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    im_mon = axes[1, 2].imshow(
        monitor, extent=extent, origin="lower", aspect="auto",
        vmin=0.0, vmax=1.0, cmap="viridis"
    )
    axes[1, 2].set_title("Monitor reward")
    fig.colorbar(im_mon, ax=axes[1, 2])

    slider_ax = fig.add_axes([0.20, 0.04, 0.60, 0.03])
    alpha_slider = Slider(
        ax=slider_ax,
        label="alpha",
        valmin=0.01,
        valmax=0.99,
        valinit=alpha0
    )

    def update(_):
        alpha = alpha_slider.val
        new_combined = interpolate_monitor_disturbance(alpha, monitor, raw_dist, X, Z)
        im_comb.set_data(new_combined)
        im_comb.set_clim(vmin=np.min(new_combined), vmax=np.max(new_combined))

        x_star, z_star, f_star = find_grid_max(X, Z, new_combined)
        norm_star = np.sqrt(x_star**2 + z_star**2)

        max_point.set_data([x_star], [z_star])
        axes[1, 1].set_title(f"Combined objective, alpha = {alpha:.2f}")
        max_text.set_text(
            f"max @ (x={x_star:.1f}, z={z_star:.1f})\n"
            f"NORM={norm_star:.1f}m\n"
            f"value={f_star:.4f}"
        )
        fig.canvas.draw_idle()

    alpha_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    plot_reward_landscape()