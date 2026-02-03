from settings import *

class Logger:
    def __init__(self):
        self.is_new = True

    def write(self, path, episode, step, entity, pos, reward, monitor_rew, dist_rew):
        # 1. Overwrite if new session, otherwise append
        mode = "w" if self.is_new else "a"
        
        with open(path, mode) as f:
            if self.is_new:
                # 2. Write custom metadata and column headers
                f.write("episode,step,entity,x,y,z,yaw,reward,monitor_rew,dist_rew\n")
                # 3. Toggle off so we don't wipe again
                self.is_new = False
            
            # 4. Standard data logging
            x, y, z, yaw = pos
            f.write(f"{episode},{step},{entity},{x},{y},{z},{yaw},{reward},{monitor_rew},{dist_rew}\n")