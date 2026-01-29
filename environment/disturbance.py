import numpy as np
import matplotlib.pyplot as plt
from environment.agents.agent import Agent

class DisturbanceField:
    """
    3D disturbance model based on:
    - Altitude (z)
    - Horizontal distance (sqrt(x² + y²))
    - Approach angle
    - Speed & acceleration gains
    """

    # -------------------------
    # Altitude component
    # -------------------------
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

    # def altitude(self, z):
    #     z = np.asarray(z, dtype=float)
    #     # Smooth decay instead of flat plateau
    #     # At z=0, val=1.0. At z=20, val=0.36. At z=100, val=0.0
    #     # The '1.0 -' ensures high penalty at low altitude
    #     # Adjust scale (e.g., 20.0) to control falloff speed
    #     return np.exp(- (z / 60.0)**2)
    
    # -------------------------
    # Horizontal distance
    # -------------------------
    def horizontal(self, d):
        d = np.asarray(d, dtype=float)
        out = np.zeros_like(d)

        out[d <= 20] = 1.0

        m1 = (d > 20) & (d <= 50)
        out[m1] = 1 - 0.2 * (d[m1] - 20) / 30

        m2 = (d > 50) & (d <= 110)
        out[m2] = 0.8 * (1 - (d[m2] - 50) / 60)

        return out

    # def horizontal(self, d):
    #     d = np.asarray(d, dtype=float)
    #     # Similar smooth decay. 
    #     # Even at d=5 vs d=10, the agent sees a difference now.
    #     return np.exp(- (d / 100.0)**2)


    # -------------------------
    # Angle gain (3D)
    # -------------------------
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


    # -------------------------
    # Motion gains
    # -------------------------
    def speed_gain(self, v, v_safe=5.0):
        return 1.0 if v <= v_safe else 1.0 + (v - v_safe) / v_safe

    def accel_gain(self, a, a_safe=2.0):
        return 1.0 if a <= a_safe else 1.0 + (a - a_safe) / a_safe


    # -------------------------
    # Disturbance between agents
    # -------------------------
    def disturbance_at(self, animal: Agent, drone: Agent):
        """
        Disturbance experienced by the animal caused by the drone
        """

        horizontal_dist = np.linalg.norm(drone.pos[0:2] - animal.pos[0:2])
        dz = abs(drone.pos[2] - animal.pos[2])

        z_alt = self.altitude(dz)
        z_hor = self.horizontal(horizontal_dist)
        g_ang = self.angle_gain(horizontal_dist, dz)

        g_speed = self.speed_gain(drone.speed)
        # g_accel = self.accel_gain(drone_accel)

        Z = z_alt * z_hor * g_ang * g_speed# * g_accel

        return float(Z)

    # -------------------------
    # Visualization
    # -------------------------
    def plot(self, cmap="tab20b"):

        X, Y, Z = self.total_field()

        plt.figure(figsize=(10, 9))

        im = plt.imshow(
            Z,
            origin="lower",
            extent=[
                -self.size, self.size,
                -self.size, self.size
            ],
            aspect="equal",
            cmap=cmap
        )

        plt.colorbar(im, label="Total Disturbance")

        plt.xlabel("Horizontal Distance (m)")
        plt.ylabel("Altitude (m)")

        plt.title("Combined Disturbance Field")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    field = DisturbanceField()
    # field.plot()

    Z = field.disturbance_at(
        x=20,
        y=20,
        v=0.0,
        a=0.0
    )

    print("Disturbance:", Z)