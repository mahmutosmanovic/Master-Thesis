import noise
import numpy as np
import matplotlib.pyplot as plt

class ResourceField:
    def __init__(self, freq, octaves, seed):
        self.freq = freq
        self.octaves = octaves
        self.base = seed

    def p_resource(self, pos):
        x, y = pos[0], pos[1]
        val = noise.pnoise2(
            x * self.freq,
            y * self.freq,
            octaves=self.octaves,
            persistence=0.5,
            lacunarity=1.5,
            base=self.base
        )
        # Normalize roughly to [0,1]
        return 0.5 * (val + 1.0)

    def get_poi(self):
        pass

    def plot_field(self, world_size=100.0, resolution=300):
        xs = np.linspace(0, world_size, resolution)
        ys = np.linspace(0, world_size, resolution)

        Z = np.zeros((resolution, resolution))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                Z[i, j] = self.p_resource((x, y))

        plt.figure(figsize=(6, 6))
        plt.imshow(
            Z.T,
            origin="lower",
            extent=[0, world_size, 0, world_size],
            interpolation="bilinear"
        )
        plt.colorbar(label="p(resource)")
        plt.title("Resource Probability Field")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    field = ResourceField(freq=0.006, octaves=2, seed=42)
    field.plot_field(world_size=1000.0, resolution=800)