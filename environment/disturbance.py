import numpy as np
import matplotlib.pyplot as plt

# Spatial Gain Functions (scalar: one animal–drone pair)
def disturbance_gain(dist_vec):
    """
    Combined disturbance for one animal-drone pair.
    """

    g_alt = altitude_gain(dist_vec)
    g_hor = horizontal_gain(dist_vec)
    g_ang = (angle_gain(dist_vec) - 1.0) / 1.0

    D = (
    0.4 * g_hor +
    0.4 * g_alt +
    0.2 * g_ang
)
    return np.clip(D, 0.0, 1.0)

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
    horizontal_dist = np.sqrt(dx*dx + dy*dy)

    theta = np.degrees(np.arctan2(z, abs(horizontal_dist)))

    if theta <= 20:
        return 1.5 - 0.5 * (theta + 90) / 110
    elif theta <= 60:
        return 1.0
    elif theta <= 90:
        return 1.0 + (theta - 60) / 30
    return 1.0


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

    COMBINED = ALT * HOR * ANG
    COMBINED = (COMBINED - COMBINED.min()) / (
        COMBINED.max() - COMBINED.min()
    )

    # FIGURE 1 — Combined disturbance
    fig1, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(X, Y, COMBINED, cmap="magma", shading="auto")
    ax.set_title("Combined 360° Disturbance Field")
    ax.set_xlabel("Horizontal Offset (m)")
    ax.set_ylabel("Altitude (m)")
    ax.plot(0, 0, "wo", markersize=8)

    fig1.colorbar(im, ax=ax, label="Normalized Intensity")
    plt.tight_layout()
    plt.savefig("../figures/disturbance_combined.png", dpi=300)


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
    plt.savefig("../figures/disturbance_components.png", dpi=300)

    plt.show()