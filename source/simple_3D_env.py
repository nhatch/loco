import pydart2 as pydart
from IPython import embed
import numpy as np

from pydart2.gui.trackball import Trackball

from stepping_stones_env import SteppingStonesEnv

SIMULATION_RATE = 1.0 / 200.0

class Simple3DEnv(SteppingStonesEnv):

    def init_camera(self):
        tb = Trackball(theta=-45.0, phi = 0.0)
        tb.trans[2] = -5 # adjust zoom
        return tb

    def step(self):
        self.world.step()
        framerate = int(1 / SIMULATION_RATE / 30)
        if self.world.frame % framerate == 0:
            print(self.world.time())
            self.render()

    def load_world(self):
        skel_file = "skel/walker3d_waist.skel"
        world = pydart.World(SIMULATION_RATE, skel_file)
        skel = world.skeletons[-1]
        skel.set_self_collision_check(True)
        return world

if __name__ == '__main__':
    env = Simple3DEnv()
    d = env.world.skeletons[1]
    q = d.q
    env.render()
    for i in range(int(5 / SIMULATION_RATE)):
        env.step()

