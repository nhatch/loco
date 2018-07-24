import pydart2 as pydart
from IPython import embed
import numpy as np

from pydart2.gui.trackball import Trackball

from stepping_stones_env import SteppingStonesEnv
from pd_control import PDController

SIMULATION_RATE = 1.0 / 2000.0

class Simple3DEnv(SteppingStonesEnv):

    def init_camera(self):
        tb = Trackball(theta=-45.0, phi=0.0, trans=[0,0,-7])
        return tb

    def step(self):
        self.world.step()
        framerate = int(1 / SIMULATION_RATE / 60)
        if self.world.frame % framerate == 0:
            print(self.world.time())
            self.render()

    def load_world(self):
        skel_file = "skel/walker3d_waist.skel"
        world = pydart.World(SIMULATION_RATE, skel_file)
        skel = world.skeletons[-1]
        skel.set_self_collision_check(True)
        self.brick_dof = 6
        return world

if __name__ == '__main__':
    env = Simple3DEnv()
    env.set_controller(PDController)
    env.reset()
    d = env.world.skeletons[1]
    q = d.q
    env.render()
    for i in range(int(2 / SIMULATION_RATE)):
        env.step()

