import matplotlib
matplotlib.use('Agg')

import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class Viewer:
    def __init__(self, dt):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        
        self.dt = dt
        self.fps = int(1 / self.dt)
        self.writer = None

    def _start_new_recording(self):
            save_dir = Path(__file__).parent.parent / "recordings"
            save_dir.mkdir(exist_ok=True)
            
            # Generates: YYYY-MM-DD_HH-MM-SS_ms
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            ms = int((time.time() % 1) * 1000)
            filename = save_dir / f"recording_{timestamp}_{ms:03d}.mp4"
            
            self.writer = FFMpegWriter(fps=self.fps)
            self.writer.setup(self.fig, str(filename), dpi=150)
            print(f"[Viewer] Started recording: {filename.name}")

    def stop_recording(self):
        if self.writer is not None:
            self.writer.finish()
            self.writer = None

    def draw(self, drones, animals, mode):
        # 1. Handle Closing: If mode is None, stop recording and exit
        if mode is None:
            if self.writer is not None:
                self.stop_recording()
            return

        # 2. Handle Opening: If mode is human but no writer exists
        if self.writer is None:
            self._start_new_recording()

        # 3. Render Frame
        self.ax.cla()
        self._setup_axes(drones, animals)

        if animals:
            a_pos = np.array([a.pos.to_numpy() for a in animals])
            self.ax.scatter(a_pos[:, 0], a_pos[:, 1], a_pos[:, 2], 
                            c="green", s=50, label="Animals", depthshade=False)

        if drones:
            d_pos = np.array([d.pos.to_numpy() for d in drones])
            self.ax.scatter(d_pos[:, 0], d_pos[:, 1], d_pos[:, 2], 
                            c="red", s=80, label="Drones", depthshade=False)
            
            for d in drones:
                p, v = d.pos.to_numpy(), d.view_dir.to_numpy()
                v = v / (np.linalg.norm(v) + 1e-8)
                self.ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], 
                               length=25, color="blue")

        self.ax.legend(loc="upper right")
        self.writer.grab_frame()

    def _setup_axes(self, drones, animals):
        all_pos = [d.pos.to_numpy() for d in drones] + [a.pos.to_numpy() for a in animals]
        if not all_pos: return
        arr = np.array(all_pos)
        low, high = np.min(arr, axis=0) - 10, np.max(arr, axis=0) + 10
        self.ax.set_xlim(low[0], high[0])
        self.ax.set_ylim(low[1], high[1])
        self.ax.set_zlim(low[2], high[2])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.stop_recording()
        plt.close(self.fig)