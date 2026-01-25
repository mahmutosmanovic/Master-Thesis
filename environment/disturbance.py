import numpy as np
import matplotlib.pyplot as plt


class DisturbanceField:
    """
    Combined disturbance model based on:
    - Altitude
    - Horizontal distance
    - Approach angle (gain field)
    """

    def __init__(self, size=100):
        self.size = size


    # -------------------------
    # Altitude component
    # -------------------------
    def altitude(self, y):

        y = np.array(y, dtype=float)

        z = np.zeros_like(y)

        m1 = (y >= 0) & (y <= 20)
        z[m1] = 1.0

        m2 = (y > 20) & (y <= 40)
        z[m2] = 1 - 0.6*(y[m2]-20)/20

        m3 = (y > 40) & (y <= 110)
        z[m3] = 0.4*(1 - (y[m3]-40)/70)

        z[y > 110] = 0

        return z


    # -------------------------
    # Horizontal distance
    # -------------------------
    def horizontal(self, x):

        x = np.array(x, dtype=float)

        z = np.zeros_like(x)

        z[x <= 20] = 1.0

        m1 = (x > 20) & (x <= 50)
        z[m1] = 1 - 0.2*(x[m1]-20)/30

        m2 = (x > 50) & (x <= 110)
        z[m2] = 0.8*(1 - (x[m2]-50)/60)

        z[x > 110] = 0

        return z


    # -------------------------
    # Angle gain
    # -------------------------
    def angle_gain(self, x, y):

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        x_sym = np.abs(x)

        theta = np.degrees(np.arctan2(y, x_sym))
        a = theta

        G = np.ones_like(a)

        # -90° – 20° : 1.5 -> 1.0
        m1 = (a >= -90) & (a <= 20)
        G[m1] = 1.5 + (1.0 - 1.5) * (a[m1] + 90) / 110

        # 20° – 60° : flat at 1.0
        m2 = (a > 20) & (a <= 60)
        G[m2] = 1.0

        # 60° – 90° : 1.0 -> 2.0
        m3 = (a > 60) & (a <= 90)
        G[m3] = 1.0 + (2.0 - 1.0) * (a[m3] - 60) / 30

        return G

    # -------------------------
    # Total disturbance
    # -------------------------
    def total_field(self):

        s = self.size

        x = np.arange(-s, s+1, 1)
        y = np.arange(-s, s+1, 1)

        X, Y = np.meshgrid(x, y)

        Z_alt = self.altitude(np.abs(Y))
        Z_hor = self.horizontal(np.abs(X))
        G_ang = self.angle_gain(X, Y)

        Z = Z_alt * Z_hor * G_ang

        return X, Y, Z
        
    def speed_gain(self, v, v_safe=5.0):

        v = max(v, 0)

        if v <= v_safe:
            G = 1
        else:
            G = 1.0 + (v - v_safe)/(v_safe)

        return G
    
    def accel_gain(self, a, a_safe=2):

        a = max(a, 0)

        if a <= a_safe:
            G = 1
        else:
            G = 1.0 + (a - a_safe)/(a_safe)

        return G
    
    def disturbance_at(self, x, y, v=0.0, a=0.0):
        """
        Disturbance at a single point with motion state
        """

        z_alt = self.altitude(abs(y))
        z_hor = self.horizontal(abs(x))
        g_ang = self.angle_gain(x, y)

        g_speed = self.speed_gain(abs(v))
        g_accel = self.accel_gain(abs(a))

        Z = z_alt * z_hor * g_ang * g_speed * g_accel

        return Z

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