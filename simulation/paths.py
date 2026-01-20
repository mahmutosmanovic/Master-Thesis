import numpy as np

class ParametricPath:
    def position(self, s):
        """
        s in R -> position (x, y, z)
        """
        raise NotImplementedError

    def tangent(self, s):
        """
        First derivative (direction of motion)
        """
        raise NotImplementedError
    
    def project_s(self, pos, s, iters=4, max_ds=0.2):
        for _ in range(iters):
            p  = self.position(s)
            dp = self.tangent(s)
            r  = p - pos
            denom = np.dot(dp, dp) + 1e-12
            ds = -np.dot(r, dp) / denom
            ds = np.clip(ds, -max_ds, max_ds)
            s += ds
        return s

class CirclePath(ParametricPath):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def position(self, s):
        return self.center + np.array([
            self.radius * np.cos(s),
            self.radius * np.sin(s),
            0.0
        ])

    def tangent(self, s):
        return np.array([
            -self.radius * np.sin(s),
             self.radius * np.cos(s),
             0.0
        ])

class FigureEightPath(ParametricPath):
    def position(self, s):
        return np.array([
            np.sin(s),
            np.sin(s) * np.cos(s),
            0.0
        ])

    def tangent(self, s):
        dx = np.cos(s)
        dy = np.cos(2*s)
        t = np.array([dx, dy, 0.0])
        return t
