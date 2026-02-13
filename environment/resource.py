import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from scipy.ndimage import maximum_filter

class ResourceField:
    def __init__(self, world_x, world_y, freq, resource_scale, resource_abundance, seed=0):
        self.world_x = world_x
        self.world_y = world_y
        self.freq = freq
        self.resource_scale = resource_scale
        self.resource_abundance = (1 - np.clip(resource_abundance, 0, 1))
        self.gen = OpenSimplex(seed)

    def p_resource(self, pos):
        x = pos[0] * self.freq
        y = pos[1] * self.freq

        val = self.gen.noise2(x, y)
        val += 0.5 * self.gen.noise2(2*x, 2*y)
        # Normalize roughly to [0,1] * resource_scale
        norm_val = 0.5 * (val + 1.0) 
        return np.clip((norm_val - self.resource_abundance) * self.resource_scale, 0, 1)

    def get_pois(self, resolution=200, kernel_size=50, threshold=1e-2):
        # Sample field
        xs = np.linspace(0, self.world_x, resolution)
        ys = np.linspace(0, self.world_y, resolution)
        Z = np.zeros((resolution, resolution))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                Z[i, j] = self.p_resource((x, y))

        # Find local maxima (kernel_size x kernel_size)
        local_max = (Z == maximum_filter(Z, size=kernel_size))

        # Threshold resource values
        peak_indices = np.argwhere(local_max & (Z >= threshold))

        # Convert to world coordinate pois
        pois = np.array([[xs[i], ys[j], Z[i, j]] for i, j in peak_indices])

        if pois.shape[0] > 0: # Sort pois in descending order
            pois = pois[np.argsort(pois[:, 2])[::-1]]

        return pois

    def plot_field(self, world_size=100.0, resolution=300):
        import os
        xs = np.linspace(0, world_size, resolution)
        ys = np.linspace(0, world_size, resolution)

        Z = np.zeros((resolution, resolution))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                Z[i, j] = self.p_resource((x, y))
        
        plt.figure(figsize=(8, 6))
        plt.imshow(
            Z.T,
            origin="lower",
            extent=[0, world_size, 0, world_size],
            interpolation="bilinear",
            cmap="CMRmap"
        )
        plt.colorbar(label="p(encounter)")

        pois = self.get_pois()
        poi_x = [x for x, _, _ in pois]
        poi_y = [y for _, y, _ in pois]
        plt.scatter(poi_x, poi_y, edgecolor="black", label="POIs")

        plt.title("Encounter Probability Field")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()

        plt.tight_layout()
        # ---- Save figure ----
        os.makedirs("figs", exist_ok=True)
        save_path = os.path.join("figs", "resource_prob.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    field = ResourceField(world_x=1000.0, world_y=1000.0, freq=0.006, resource_scale=0.35, resource_abundance=0.4, seed=1234567)
    field.plot_field(world_size=1000.0, resolution=200)