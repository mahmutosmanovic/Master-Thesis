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
        t = np.array([
            -np.sin(s),
             np.cos(s),
             0.0
        ])
        return t / np.linalg.norm(t)

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
        return t / np.linalg.norm(t)
