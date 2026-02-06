import random
import math

class Vector:
    def __init__(self, x=0, y=0, z=0, 
                 random_unit_2d=False,
                 random_unit_3d=False):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        if random_unit_2d:
            self.random_unit_2d()
        elif random_unit_3d:
            self.random_unit_3d()

    def __repr__(self):
        return f"[{self.x}, {self.y}, {self.z}]"

    def norm(self, order=2):
        return (self.x**order + self.y**order + self.z**order)**(1/order)

    def unit(self):
        n = self.norm()
        self.x /= n 
        self.y /= n 
        self.z /= n
    
    def random_unit_2d(self):
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.z = float(0)
        self.unit()

    def random_unit_3d(self):
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.z = random.uniform(-1, 1)
        self.unit()
        
    def scale(self, k):
        self.x *= k
        self.y *= k
        self.z *= k
    
    def add(self, vec3d):
        self.x += vec3d.x
        self.y += vec3d.y
        self.z += vec3d.z

    def setter(self, vec3d):
        self.x = vec3d.x
        self.y = vec3d.y
        self.z = vec3d.z

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

