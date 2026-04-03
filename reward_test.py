import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Geometry helpers
def angle_gain_grid(X, Z):
    horizontal_dist = np.abs(X)
    a = np.degrees(np.arctan2(Z, horizontal_dist)) % 360.0

    G = np.zeros_like(a, dtype=float)

    m1 = a < 20.0
    t1 = (a[m1] + 90.0) / 110.0
    G[m1] = 0.5 * (1.0 - t1)

    m2 = (a >= 20.0) & (a < 60.0)
    G[m2] = 0.0

    m3 = (a >= 60.0) & (a < 90.0)
    G[m3] = (a[m3] - 60.0) / 30.0

    m4 = (a >= 90.0) & (a < 120.0)
    G[m4] = 1.0 - (a[m4] - 90.0) / 30.0

    m5 = (a >= 120.0) & (a < 160.0)
    G[m5] = 0.0

    m6 = (a >= 160.0) & (a < 270.0)
    G[m6] = 0.5 * (a[m6] - 160.0) / 110.0

    m7 = a >= 270.0
    t7 = (a[m7] - 270.0) / 110.0
    G[m7] = 0.5 * (1.0 - t7)

    return G

def animal_axis_gain_xz_grid(X, animal_vel_dir):
    a = np.asarray(animal_vel_dir, dtype=float)
    a_xy = a[:2]
    a_norm = np.linalg.norm(a_xy)

    if a_norm < 1e-8:
        return np.zeros_like(X, dtype=float)

    const_val = abs(a_xy[0]) / a_norm
    G = np.full_like(X, const_val, dtype=float)
    G[np.abs(X) < 1e-12] = 0.0
    return G

# Monitoring penalty and low-disturbance utility
def monitoring_penalty_grid(X, Z, r_f):
    d = np.sqrt(X * X + Z * Z)
    return -d / max(r_f, 1e-9)

def low_disturbance_utility_grid(X,Z,animal_vel_dir,R,Z_scale,w_angle,w_axis):
    """
    Recommended formulation:

      s0 = r/R + |z|/Z
      g  = exp(-s0)

      angle_eff = angle_bad * g
      axis_eff  = axis_bad  * g

      s = (r/R) * (1 + w_axis * axis_eff) + |z|/Z

      U_base = 1 - exp(-s)

      U = U_base * (1 - w_angle * angle_eff) * (1 - w_axis * axis_eff)

    Properties:
      - angle/axis effects decay with distance
      - axis both lowers reward and pushes optimum outward
      - utility stays in [0, 1]
    """
    radial_distance = np.abs(X)   # x-z slice => horizontal distance is |x|
    z_distance = np.abs(Z)

    angle_bad = angle_gain_grid(X, Z)
    axis_bad = animal_axis_gain_xz_grid(X, animal_vel_dir)

    # Base distance score
    s0 = (
        radial_distance / max(R, 1e-9) +
        z_distance / max(Z_scale, 1e-9)
    )

    # Distance gate: geometry matters nearby, fades far away
    g = np.exp(-s0)

    angle_eff = angle_bad * g
    axis_eff = axis_bad * g

    # Axis pushes preferred stand-off outward
    s = (
        (radial_distance / max(R, 1e-9)) * (1.0 + w_axis * axis_eff) +
        z_distance / max(Z_scale, 1e-9)
    )

    U_base = 1.0 - np.exp(-s)

    # Geometry also reduces amplitude
    f_angle = np.clip(1.0 - w_angle * angle_eff, 0.0, 1.0)
    f_axis = np.clip(1.0 - w_axis * axis_eff, 0.0, 1.0)

    U = U_base * f_angle * f_axis
    return np.clip(U, 0.0, 1.0)

def total_reward_raw_grid( X, Z, animal_vel_dir, R, Z_scale, w_angle, w_axis, r_f, d_s):
    P = monitoring_penalty_grid(X, Z, r_f=r_f)
    U = low_disturbance_utility_grid(
        X=X,
        Z=Z,
        animal_vel_dir=animal_vel_dir,
        R=R,
        Z_scale=Z_scale,
        w_angle=w_angle,
        w_axis=w_axis,
    )
    return (1.0 - d_s) * P + d_s * U

# Analytic base-only maximum (radial + z only)
def analytic_base_case_max(R, Z_scale, r_f, d_s, eps=1e-12):
    """
    Base-only reward:

        R_base(x,z) =
            (1-d_s) * (-sqrt(x^2+z^2) / r_f)
            + d_s * (1 - exp(-(x/R + z/Z_scale)))

    over x,z >= 0.

    Returns:
        base_max, x_star, z_star
    """
    if d_s <= eps:
        return 0.0, 0.0, 0.0

    if (1.0 - d_s) <= eps:
        return 1.0, np.inf, np.inf

    S = np.sqrt(R * R + Z_scale * Z_scale)

    log_arg = (
        d_s * r_f * S
        / max((1.0 - d_s) * R * Z_scale, eps)
    )

    if log_arg <= 1.0:
        return 0.0, 0.0, 0.0

    lam = (R * Z_scale / (R * R + Z_scale * Z_scale)) * np.log(log_arg)

    x_star = lam * Z_scale
    z_star = lam * R

    d = lam * S
    expo = lam * (R * R + Z_scale * Z_scale) / (R * Z_scale)

    base_max = (
        (1.0 - d_s) * (-d / r_f)
        + d_s * (1.0 - np.exp(-expo))
    )

    return float(base_max), float(x_star), float(z_star)


def evaluate_grids( X, Z, animal_vel_dir, R, Z_scale, w_angle, w_axis, r_f, d_s):
    P = monitoring_penalty_grid(X, Z, r_f=r_f)
    U = low_disturbance_utility_grid(
        X=X,
        Z=Z,
        animal_vel_dir=animal_vel_dir,
        R=R,
        Z_scale=Z_scale,
        w_angle=w_angle,
        w_axis=w_axis,
    )
    T_raw = (1.0 - d_s) * P + d_s * U

    # Base max still computed ONLY from radial + vertical terms
    base_max, x_base_star, z_base_star = analytic_base_case_max(
        R=R,
        Z_scale=Z_scale,
        r_f=r_f,
        d_s=d_s,
    )

    if base_max <= 1e-12:
        T_final = np.zeros_like(T_raw)
    else:
        T_final = np.clip(T_raw / base_max, 0.0, 1.0)

    return P, U, T_raw, T_final, base_max, x_base_star, z_base_star



# ------------------------------------------------------------
# Main interactive plot
# ------------------------------------------------------------
def main():
    animal_vel_dir = np.array([1.0, 0.0, 0.0], dtype=float)

    x = np.linspace(-200.0, 200.0, 180)
    z = np.linspace(-200.0, 200.0, 180)
    X, Z = np.meshgrid(x, z)

    R0 = 100.0
    Z0 = 60.0
    angle0 = 0.5
    axis0 = 0.7
    ds0 = 0.5
    rf0 = 200.0

    P0, U0, Traw0, Tfinal0, base_max0, xbs0, zbs0 = evaluate_grids(
        X=X,
        Z=Z,
        animal_vel_dir=animal_vel_dir,
        R=R0,
        Z_scale=Z0,
        w_angle=angle0,
        w_axis=axis0,
        r_f=rf0,
        d_s=ds0,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.30, wspace=0.20)

    ax_p, ax_u, ax_traw, ax_tfinal = axes

    im_p = ax_p.imshow(
        P0,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto"
    )
    im_u = ax_u.imshow(
        U0,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0
    )
    im_traw = ax_traw.imshow(
        Traw0,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        vmax=1.0
    )
    im_tfinal = ax_tfinal.imshow(
        Tfinal0,
        extent=[x.min(), x.max(), z.min(), z.max()],
        origin="lower",
        aspect="auto",
        vmax=1.0
    )

    ax_p.set_title("Monitoring penalty")
    ax_u.set_title("Low-disturbance utility")
    ax_traw.set_title("Total reward raw")
    ax_tfinal.set_title(f"Total reward final\nbase max={base_max0:.3f}, d_s={ds0:.3f}")

    for ax in axes:
        ax.set_xlabel("Horizontal distance")
        ax.set_ylabel("Vertical distance")
        ax.axhline(0.0, linewidth=0.8, alpha=0.4)
        ax.axvline(0.0, linewidth=0.8, alpha=0.4)

    if np.isfinite(xbs0) and np.isfinite(zbs0):
        p1, = ax_traw.plot([xbs0], [zbs0], "wo", markersize=6)
        p2, = ax_tfinal.plot([xbs0], [zbs0], "wo", markersize=6)
    else:
        p1, = ax_traw.plot([], [], "wo", markersize=6)
        p2, = ax_tfinal.plot([], [], "wo", markersize=6)

    fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04).set_label("Penalty")
    fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04).set_label("Utility")
    fig.colorbar(im_traw, ax=ax_traw, fraction=0.046, pad=0.04).set_label("Raw total")
    fig.colorbar(im_tfinal, ax=ax_tfinal, fraction=0.046, pad=0.04).set_label("Final total")

    # Sliders
    ax_R = plt.axes([0.12, 0.20, 0.76, 0.025])
    ax_Z = plt.axes([0.12, 0.16, 0.76, 0.025])
    ax_angle = plt.axes([0.12, 0.12, 0.76, 0.025])
    ax_axis = plt.axes([0.12, 0.08, 0.76, 0.025])
    ax_ds = plt.axes([0.12, 0.04, 0.76, 0.025])

    s_R = Slider(ax_R, "R", 10.0, 300.0, valinit=R0)
    s_Z = Slider(ax_Z, "Z", 10.0, 200.0, valinit=Z0)
    s_angle = Slider(ax_angle, "angle", 0.0, 1.0, valinit=angle0)
    s_axis = Slider(ax_axis, "axis", 0.0, 1.0, valinit=axis0)
    s_ds = Slider(ax_ds, "d_s", 0.0, 1.0, valinit=ds0)

    def update(_):
        Pm, U, Traw, Tfinal, base_max, xbs, zbs = evaluate_grids(
            X=X,
            Z=Z,
            animal_vel_dir=animal_vel_dir,
            R=s_R.val,
            Z_scale=s_Z.val,
            w_angle=s_angle.val,
            w_axis=s_axis.val,
            r_f=rf0,
            d_s=s_ds.val,
        )

        im_p.set_data(Pm)
        im_u.set_data(U)
        im_traw.set_data(Traw)
        im_tfinal.set_data(Tfinal)

        ax_tfinal.set_title(
            f"Total reward final\nbase max={base_max:.3f}, d_s={s_ds.val:.3f}"
        )

        if np.isfinite(xbs) and np.isfinite(zbs):
            p1.set_data([xbs], [zbs])
            p2.set_data([xbs], [zbs])
        else:
            p1.set_data([], [])
            p2.set_data([], [])

        im_p.set_clim(np.min(Pm), 0.0)
        im_u.set_clim(0.0, 1.0)
        im_traw.set_clim(np.min(Traw), 1.0)
        im_tfinal.set_clim(np.min(Tfinal), 1.0)

        fig.canvas.draw_idle()

    for slider in (s_R, s_Z, s_angle, s_axis, s_ds):
        slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()