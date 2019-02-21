from IPython import embed
import numpy as np
from pd_control import PDController

class ReplayController(PDController):
    def __init__(self, env):
        super().__init__(env)
        self.i = 0

    def compute(self):
        c = self.env.consts()
        if self.env.world.frame % c.FRAMES_PER_CONTROL == 0:
            self.target_q = self.trajectory[self.i]
            self.i += 1
        return self.compute_PD()

    def load(self, filename):
        self.trajectory = np.loadtxt(filename)

if __name__ == "__main__":
    from darwin_env import DarwinEnv
    env = DarwinEnv(ReplayController)
    env.controller.load("data/States.txt")
    c = env.consts()
    env.reset(random=0.005, render=1)
    env.sdf_loader.put_grounds([[-1, c.GROUND_LEVEL, 0]])
    env.zoom = 1
    env.track_point = [0.2,0,-0.1]
    env.run(6)
