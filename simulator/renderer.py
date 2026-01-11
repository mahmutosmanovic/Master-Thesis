from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


class Renderer3D:
    def __init__(self, world, title: str = "World (3D)"):
        self.world = world
        self.title = title

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(self.title)

        lo = np.asarray(world.bounds_min, dtype=float)
        hi = np.asarray(world.bounds_max, dtype=float)

        self.ax.set_xlim(lo[0], hi[0])
        self.ax.set_ylim(lo[1], hi[1])
        self.ax.set_zlim(lo[2], hi[2])

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.s_animals = self.ax.scatter([], [], [], marker="o", label="animals")
        self.s_drones = self.ax.scatter([], [], [], marker="^", label="drones")
        self.s_other = self.ax.scatter([], [], [], marker="s", label="other")

        self.ax.legend(loc="upper right")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self, t: int | None = None):
        animals, drones, other = [], [], []
        for a in self.world.agents:
            p = np.asarray(a.state.pos, dtype=float).reshape(3)
            kind = getattr(a.params, "agent_type", "other")
            if kind == "animal":
                animals.append(p)
            elif kind == "drone":
                drones.append(p)
            else:
                other.append(p)

        def _set(scatter, pts):
            if len(pts) == 0:
                scatter._offsets3d = ([], [], [])
                return
            P = np.vstack(pts)
            scatter._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

        _set(self.s_animals, animals)
        _set(self.s_drones, drones)
        _set(self.s_other, other)

        if t is not None:
            self.ax.set_title(f"{self.title} | t={t}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
