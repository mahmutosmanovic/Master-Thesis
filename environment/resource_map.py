import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from scipy.ndimage import maximum_filter
from .vec import Vector

class ResourceMap:
    def __init__(self, config, seed=0):
        self.gen = OpenSimplex(seed)

        self.p_wavelenght = config.resource.p_wavelenght
        self.p_reduction = config.resource.p_reduction
        self.p_scale = config.resource.p_scale
        self.sample_resolution = config.resource.sample_res
        self.threshold = config.resource.min_poi_p
        self.world_size = config.animal.init.max_spawn_radius*2

        self.kernel_size = config.resource.kernel_size//config.resource.sample_res
        # odd values are better, otherwise kernel can shift values (not same distance on each side of center)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.generate_pois()
    
    def p(self, pos):
        x = pos.x / self.p_wavelenght
        y = pos.y / self.p_wavelenght

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
        x_space = np.arange(-self.world_size/2, self.world_size/2, self.sample_resolution)
        y_space = np.arange(-self.world_size/2, self.world_size/2, self.sample_resolution)
        p_map = np.zeros((len(x_space), len(x_space)))

        for i, x in enumerate(x_space):
            for j, y in enumerate(y_space):
                p_map[i, j] = self.p(Vector(x=x, y=y))

        return p_map, x_space, y_space
    
    def get_pois(self):
        return self.pois
    
    def generate_pois(self):
        p_map, x_space, y_space = self.sample_map()

        # Find local maxima in kernel
        local_maxima = (p_map == maximum_filter(p_map, size=self.kernel_size))

        # Threshold resource values
        peak_indices = np.argwhere(local_maxima & (p_map >= self.threshold))

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
    from scripts.config import cfg_train

    rm = ResourceMap(Box(cfg_train), 0)

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