import math
import random
import numpy as np

class Vector:
    def __init__(self, x=0, y=0, z=0,
                 random_unit_2d=False,
                 random_unit_3d=False,
                 rng=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        if random_unit_2d and rng is not None:
            self.random_unit_2d(rng)
        elif random_unit_3d and rng is not None:
            self.random_unit_3d(rng)

    def __repr__(self):
        return f"[{self.x}, {self.y}, {self.z}]"

    def norm(self, order=2):
        return (self.x**order + self.y**order + self.z**order)**(1/order)

    def unit(self):
        n = self.norm()
        self.x /= n 
        self.y /= n 
        self.z /= n

    def distance(self, vec_other):
        dist_vec = self.add(vec_other.scale(-1))
        return dist_vec.norm()

    def random_unit_2d(self, rng):
        self.x = rng.uniform(-1, 1)
        self.y = rng.uniform(-1, 1)
        self.z = float(0)
        self.unit()

    def random_unit_3d(self, rng):
        self.x = rng.uniform(-1, 1)
        self.y = rng.uniform(-1, 1)
        self.z = rng.uniform(-1, 1)
        self.unit()

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])
    
    def rotate_z(self, deg):
        rad = math.radians(deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        # save old values
        x = self.x
        y = self.y

        # rotate around z-axis
        self.x = x * cos_t - y * sin_t
        self.y = x * sin_t + y * cos_t
    
    def scale(self, k, in_place=False):
        if in_place:
            self.x *= k
            self.y *= k
            self.z *= k
        else:
            x = self.x * k
            y = self.y * k
            z = self.z * k
            return Vector(x, y, z)
        
    def add(self, vec3d, in_place=False):
        if in_place:
            self.x += vec3d.x
            self.y += vec3d.y
            self.z += vec3d.z
        else:
            x = self.x + vec3d.x
            y = self.y + vec3d.y
            z = self.z + vec3d.z
            return Vector(x, y, z)

    def setter(self, vec3d):
        self.x = vec3d.x
        self.y = vec3d.y
        self.z = vec3d.z

    def getter(self):
        return Vector(self.x, self.y, self.z)



