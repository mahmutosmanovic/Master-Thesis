from settings import *

# ======================================================
# Disturbance Model
# ======================================================

class DisturbanceField:
    """
    Disturbance = altitude_gain(|dz|) * horizontal_gain(sqrt(dx^2+dy^2)) * angle_gain(horiz, dz)
    Returned disturbance is normalized to [0, 1] by dividing by the theoretical max of angle_gain (=2).
    """

    # -------------------------
    # Altitude Gain
    # -------------------------
    def altitude_gain(self, z):
        z = np.asarray(z, dtype=float)
        out = np.zeros_like(z)

        m1 = (z >= 0) & (z <= 20)
        out[m1] = 1.0

        m2 = (z > 20) & (z <= 40)
        out[m2] = 1.0 - 0.6 * (z[m2] - 20.0) / 20.0

        m3 = (z > 40) & (z <= 110)
        out[m3] = 0.4 * (1.0 - (z[m3] - 40.0) / 70.0)

        # z > 110 -> 0 (already default)
        return out


    # -------------------------
    # Horizontal Gain
    # -------------------------
    def horizontal_gain(self, d):
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)

        m1 = (d >= 0) & (d <= 20)
        out[m1] = 1.0

        m2 = (d > 20) & (d <= 50)
        out[m2] = 1.0 - 0.2 * (d[m2] - 20.0) / 30.0

        m3 = (d > 50) & (d <= 110)
        out[m3] = 0.8 * (1.0 - (d[m3] - 50.0) / 60.0)

        # d > 110 -> 0 (already default)
        return out


    # -------------------------
    # Angle Gain (YOUR EXACT RULE)
    # theta = atan2(z, horiz) in degrees, range [-90, +90]
    #
    # 90 -> 60  : decrease 2 -> 1
    # 60 -> 20  : flat at 1
    # 20 -> -90 : increase 1 -> 1.5
    # -------------------------
    def angle_gain(self, horiz, z):
        horiz = np.asarray(horiz, dtype=float)
        z = np.asarray(z, dtype=float)

        # Avoid division issues at horiz=0 (atan2 handles it fine, but keep consistent dtype)
        theta = np.degrees(np.arctan2(z, horiz))

        G = np.ones_like(theta)

        # 90 -> 60 : 2 -> 1 (linear)
        m1 = (theta >= 60.0) & (theta <= 90.0)
        # at 60 => 1, at 90 => 2
        G[m1] = 1.0 + (theta[m1] - 60.0) / 30.0

        # 60 -> 20 : flat at 1
        m2 = (theta >= 20.0) & (theta < 60.0)
        G[m2] = 1.0

        # 20 -> -90 : 1 -> 1.5 (linear)
        m3 = (theta >= -90.0) & (theta < 20.0)
        # at 20 => 1.0, at -90 => 1.5
        G[m3] = 1.0 + (20.0 - theta[m3]) / 110.0 * 0.5

        return G


    # -------------------------
    # Normalized Disturbance (0..1)
    # animal, drone: tuples (x, y, z)
    # -------------------------
    def get_disturbance(self, animal, drone):
        xa, ya, za = animal
        xd, yd, zd = drone

        dx = xd - xa
        dy = yd - ya
        dz = zd - za

        horiz = np.sqrt(dx**2 + dy**2)
        vert = np.abs(dz)

        g_alt = self.altitude_gain(vert)
        g_hor = self.horizontal_gain(horiz)
        g_ang = self.angle_gain(horiz, dz)

        raw = g_alt * g_hor * g_ang

        # Normalize by max possible angle gain (2.0)
        return float(np.clip(raw / 2.0, 0.0, 1.0))


# ======================================================
# 3-View Visualization (XY, XZ, YZ)
# ======================================================

def plot_3view_contours(df, extent=50.0, n=120, levels=30):
    """
    Shows 3 orthogonal slices through the 3D disturbance volume at the center:
      - XY at Z=0
      - XZ at Y=0
      - YZ at X=0
    """

    vals = np.linspace(-extent, extent, n)
    X, Y, Z = np.meshgrid(vals, vals, vals, indexing="ij")

    horiz = np.sqrt(X**2 + Y**2)
    vert = np.abs(Z)

    g_alt = df.altitude_gain(vert)
    g_hor = df.horizontal_gain(horiz)
    g_ang = df.angle_gain(horiz, Z)  # dz = Z (signed)

    D = np.clip((g_alt * g_hor * g_ang) / 2.0, 0.0, 1.0)

    mid = n // 2

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(right=0.88)

    cmap = "tab20b"

    # IMPORTANT:
    # With indexing="ij", slices come out as [x,y] / [x,z] / [y,z]
    # but contourf expects array indexed as [vertical_axis, horizontal_axis]
    # so we transpose each slice.

    # XY (Z=0): slice is [x, y] -> transpose to [y, x]
    im1 = axs[0].contourf(vals, vals, D[:, :, mid].T, levels=levels, cmap=cmap)
    axs[0].set_title("XY Plane (Z = 0)")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect("equal")

    # XZ (Y=0): slice is [x, z] -> transpose to [z, x]
    im2 = axs[1].contourf(vals, vals, D[:, mid, :].T, levels=levels, cmap=cmap)
    axs[1].set_title("XZ Plane (Y = 0)")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    axs[1].set_aspect("equal")

    # YZ (X=0): slice is [y, z] -> transpose to [z, y]
    im3 = axs[2].contourf(vals, vals, D[mid, :, :].T, levels=levels, cmap=cmap)
    axs[2].set_title("YZ Plane (X = 0)")
    axs[2].set_xlabel("Y")
    axs[2].set_ylabel("Z")
    axs[2].set_aspect("equal")

    # Colorbar outside
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im3, cax=cax, label="Disturbance")

    fig.suptitle("3-View Disturbance Field Slices", fontsize=16)
    plt.savefig("disturbance")
    plt.show()

# ======================================================
# Main
# ======================================================

def main():
    df = DisturbanceField()
    plot_3view_contours(df, extent=50.0, n=120, levels=30)


if __name__ == "__main__":
    main()
