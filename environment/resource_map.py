import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from scipy.ndimage import maximum_filter

class ResourceMap:
    def __init__(self, p_wavelenght, p_reduction, p_scale, sample_res, min_poi_p, world_size, kernel_size, seed=0):
        self.gen = OpenSimplex(seed)

        self.p_wavelenght = p_wavelenght
        self.p_reduction = p_reduction
        self.p_scale = p_scale
        self.sample_resolution = sample_res
        self.min_poi_p = min_poi_p
        self.world_size = world_size

        self.kernel_size = kernel_size//sample_res
        # odd values are better, otherwise kernel can shift values (not same distance on each side of center)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.generate_pois()
    
    def p(self, pos):
        x = pos[0] / self.p_wavelenght
        y = pos[1] / self.p_wavelenght

        # two octave perlin noise
        val  = self.gen.noise2(x, y) # [-1, 1]
        val += self.gen.noise2(2*x, 2*y) * 0.5  # [-1, 1] + [-0.5, 0.5] -> [-1.5, 1.5]
        p_raw = (val + 1.5) / 3  # [-1.5, 1.5] -> [0, 1]
        transformed_val = (p_raw * self.p_scale) - self.p_reduction
        return np.clip(transformed_val, 0, 1)

    def is_encounter(self, pos, rng):
        p_encounter = self.p(pos)
        return rng.choice([True, False], p=[p_encounter, 1-p_encounter]), p_encounter # -> [encounter yes/no, probability of encounter]
    
    def sample_map(self):
        x_space = np.arange(0, self.world_size, self.sample_resolution)
        y_space = np.arange(0, self.world_size, self.sample_resolution)
        p_map = np.zeros((len(x_space), len(x_space)))

        for i, x in enumerate(x_space):
            for j, y in enumerate(y_space):
                p_map[i, j] = self.p([x, y])

        return p_map, x_space, y_space
    
    def get_pois(self):
        return self.pois
    
    def generate_pois(self):
        p_map, x_space, y_space = self.sample_map()

        # Find local maxima in kernel
        local_maxima = (p_map == maximum_filter(p_map, size=self.kernel_size))

        # Threshold resource values
        peak_indices = np.argwhere(local_maxima & (p_map >= self.min_poi_p))

        # Convert to world coordinate poi array
        pois = np.array([[x_space[i], y_space[j], p_map[i, j]] for i, j in peak_indices])

        if pois.shape[0] > 0: # Sort pois in descending order, highest probability first
            pois = pois[np.argsort(pois[:, 2])[::-1]]

        self.pois = pois
    
    def reset(self, seed):
        self.gen = OpenSimplex(seed)


def plot_resource_map():
    import os
    from box import Box

    rm = ResourceMap(
        p_wavelenght=125,
        p_reduction=0.2,
        p_scale=0.5,
        sample_res=10,
        min_poi_p=1e-2,
        world_size=1000,
        kernel_size=150,
        seed=0)

    p_map, x_space, y_space = rm.sample_map()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(
        p_map.T,
        origin="lower",
        extent=[min(x_space), max(x_space), min(y_space), max(y_space)],
        interpolation="bilinear",
        cmap="CMRmap"
    )
    plt.colorbar(label="p(encounter)")

    pois = rm.get_pois()
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

if __name__ == "__main__":
    plot_resource_map()