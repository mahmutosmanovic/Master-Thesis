from settings import *

class Logger:
    def __init__(self):
        self.is_new = True

    def write(self, path, t, entity, pos):
        # 1. Overwrite if new session, otherwise append
        mode = "w" if self.is_new else "a"
        
        with open(path, mode) as f:
            if self.is_new:
                # 2. Write custom metadata and column headers
                f.write("t,entity,x,y,z,yaw\n")
                # 3. Toggle off so we don't wipe again
                self.is_new = False
            
            # 4. Standard data logging
            x, y, z, yaw = pos
            f.write(f"{t},{entity},{x},{y},{z},{yaw}\n")