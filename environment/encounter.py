import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from scipy.ndimage import maximum_filter

class EncounterMap:
    def __init__(self, config, seed=0):
        self.gen = OpenSimplex(seed)

        self.freq = config.encounter.p_freq
        self.p_reduction = config.encounter.p_reduction
        self.p_scale = config.encounter.p_scale
        self.sample_resolution = config.encounter.sample_res
        self.kernel_size = config.encounter.kernel_size
        self.threshold = config.encounter.min_poi_p
        self.world_size = config.init.max_spawn_radius*2

        self.pois = self.generate_pois()

    def p_encounter(self, pos):
        x = pos.x * self.freq
        y = pos.y * self.freq

        # two octave perlin noise
        val  = self.gen.noise2(x, y) # [-1, 1]
        val += self.gen.noise2(2*x, 2*y) * 0.5  # [-1, 1] + [-0.5, 0.5] -> [-1.5, 1.5]
        p_raw = (val + 1.5) / 3  # [-1.5, 1.5] -> [0, 1]
        transformed_val = (p_raw - self.p_reduction) * self.p_scale
        return np.clip(transformed_val, 0, 1)

    def is_encounter(self, pos, rng):
        p_encounter = self.p_encounter(pos)
        return rng.choice([True, False], p=[p_encounter, 1-p_encounter]), p_encounter # -> [encounter yes/no, probability of encounter]
    
    def sample_map(self):
        x_space = np.linspace(self.world_size/2, self.world_size/2, self.sample_resolution)
        y_space = np.linspace(self.world_size/2, self.world_size/2, self.sample_resolution)
        p_map = np.zeros((self.sample_resolution, self.sample_resolution))

        for i, x in enumerate(x_space):
            for j, y in enumerate(y_space):
                p_map[i, j] = self.p_encounter((x, y))

        return p_map, x_space, y_space
    
    def get_pois(self):
        return self.poi_cache
    
    def generate_pois(self):
        p_map, x_space, y_space = self.sample_map(self.world_size, self.sample_resolution)

        # Find local maxima in kernel
        local_max = (p_map == maximum_filter(p_map, size=self.kernel_size))

        # Threshold resource values
        peak_indices = np.argwhere(local_max & (p_map >= self.threshold))

        # Convert to world coordinate poi array
        pois = np.array([[x_space[i], y_space[j], p_map[i, j]] for i, j in peak_indices])

        if pois.shape[0] > 0: # Sort pois in descending order, highest probability first
            pois = pois[np.argsort(pois[:, 2])[::-1]]

        return pois

def plot_encounter_p():
    pass