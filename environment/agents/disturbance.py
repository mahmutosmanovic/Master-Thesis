import numpy as np
import matplotlib.pyplot as plt
from environment.agents.agent import Agent

class DisturbanceField:
    def altitude(self, z):
        z = np.asarray(z, dtype=float)
        out = np.zeros_like(z)

        m1 = (z >= 0) & (z <= 20)
        out[m1] = 1.0

        m2 = (z > 20) & (z <= 40)
        out[m2] = 1 - 0.6 * (z[m2] - 20) / 20

        m3 = (z > 40) & (z <= 110)
        out[m3] = 0.4 * (1 - (z[m3] - 40) / 70)

        return out

    def horizontal(self, d):
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)

        out[d <= 20] = 1.0

        m1 = (d > 20) & (d <= 50)
        out[m1] = 1 - 0.2 * (d[m1] - 20) / 30

        m2 = (d > 50) & (d <= 110)
        out[m2] = 0.8 * (1 - (d[m2] - 50) / 60)

        return out

    def angle_gain(self, horizontal_dist, z):
        """
        Angle between horizontal plane and line of sight
        """
        theta = np.degrees(np.arctan2(z, horizontal_dist))
        G = np.ones_like(theta)

        # -90° – 20°
        m1 = (theta >= -90) & (theta <= 20)
        G[m1] = 1.5 + (1.0 - 1.5) * (theta[m1] + 90) / 110

        # 20° – 60°
        m2 = (theta > 20) & (theta <= 60)
        G[m2] = 1.0

        # 60° – 90°
        m3 = (theta > 60) & (theta <= 90)
        G[m3] = 1.0 + (2.0 - 1.0) * (theta[m3] - 60) / 30

        return G

    def heading_gain_hard(self, drone_vel, diff):
        """
        Gain:
        2.0 -> straight toward
        1.0 -> parallel
        1.0 -> straight away
        """
        v = np.asarray(drone_vel, dtype=float)
        d = np.asarray(diff, dtype=float)

        v_norm = np.linalg.norm(v)
        d_norm = np.linalg.norm(d)

        if v_norm < 1e-8 or d_norm < 1e-8:
            return 1.0

        cos_theta = np.dot(v, d) / (v_norm * d_norm)

        return 1.0 + max(0.0, cos_theta)

    def heading_gain_soft(self, drone_vel, diff):
        """
        Gain:
        2.0 -> straight toward
        1.5 -> parallel
        1.0 -> straight away
        """
        v = np.asarray(drone_vel, dtype=float)
        d = np.asarray(diff, dtype=float)

        v_norm = np.linalg.norm(v)
        d_norm = np.linalg.norm(d)

        if v_norm < 1e-8 or d_norm < 1e-8:
            return 1.5

        cos_theta = np.dot(v, d) / (v_norm * d_norm)

        return 1.5 + 0.5 * cos_theta

    def speed_gain(self, v, v_safe=5.0):
        return 1.0 if v <= v_safe else 1.0 + (v - v_safe) / v_safe

    def get_disturbance(self, animal: Agent, drone: Agent):
        diff = drone.pos - animal.pos
        horizontal_dist = np.linalg.norm(diff[0:2])
        dz_abs = abs(diff[2])
        dz = diff[2]

        z_alt = self.altitude(dz_abs)
        z_hor = self.horizontal(horizontal_dist)
        g_ang = self.angle_gain(horizontal_dist, dz)

        g_speed = self.speed_gain(drone.norm_speed * drone.params.max_speed)

        # g_heading = self.heading_gain_soft(drone.direction, diff)
        g_heading = self.heading_gain_hard(drone.direction, diff)
        # g_accel = self.accel_gain(drone_accel)

        Z = z_alt * z_hor * g_ang * g_speed * g_heading # * g_accel

        return {"val": float(Z), "dir": diff}

def plot_3view_contours(df, extent=50.0, n=120, levels=30):
    vals = np.linspace(-extent, extent, n)
    X, Y, Z = np.meshgrid(vals, vals, vals, indexing="ij")

    horizontal_dist = np.sqrt(X**2 + Y**2)
    dz = abs(Z)

    z_alt = df.altitude(dz)
    z_hor = df.horizontal(horizontal_dist)
    g_ang = df.angle_gain(horizontal_dist, Z)

    D = np.clip(z_alt * z_hor * g_ang, 0.0, 2.0)

    mid = n // 2

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(right=0.88)

    cmap = "tab20b"

    im1 = axs[0].contourf(vals, vals, D[:, :, mid].T, levels=levels, cmap=cmap)
    axs[0].set_title("XY Plane (Z = 0)")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect("equal")

    im2 = axs[1].contourf(vals, vals, D[:, mid, :].T, levels=levels, cmap=cmap)
    axs[1].set_title("XZ Plane (Y = 0)")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    axs[1].set_aspect("equal")

    im3 = axs[2].contourf(vals, vals, D[mid, :, :].T, levels=levels, cmap=cmap)
    axs[2].set_title("YZ Plane (X = 0)")
    axs[2].set_xlabel("Y")
    axs[2].set_ylabel("Z")
    axs[2].set_aspect("equal")

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