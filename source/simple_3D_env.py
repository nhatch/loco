import pydart2 as pydart
from IPython import embed
import numpy as np

from pydart2.gui.trackball import Trackball, _q_add

from stepping_stones_env import SteppingStonesEnv
import consts_3D
from state_3D import State3D

SIMULATION_RATE = 1.0 / 2000.0

class Simple3DEnv(SteppingStonesEnv):
    def update_viewer(self, com):
        angle = np.pi/6
        dist = com[0]
        t1 = np.sin(angle)*dist
        t2 = np.cos(angle)*dist + 5
        self.viewer.scene.tb.trans[2] = -t2
        self.viewer.scene.tb.trans[1] = t1

    def init_camera(self):
        # The default Trackball parameters rotate phi around the z axis
        # rather than the y axis. I modified this code from the
        # Trackball.get_orientation method.
        theta = -np.pi/6
        phi = np.pi / 2
        xrot = np.array([np.sin(theta/2), 0, 0, np.cos(theta/2)])
        yrot = np.array([0, np.sin(phi/2), 0, np.cos(phi/2)])
        rot = _q_add(xrot, yrot)

        tb = Trackball(rot=rot, trans=[0,0,-5])
        return tb

    def step(self, frames_per_second=60):
        self.world.step()
        framerate = int(1 / SIMULATION_RATE / frames_per_second)
        if self.world.frame % framerate == 0:
            print(self.world.time())
            self.render()

    def wrap_state(self, raw_state):
        return State3D(raw_state)

    def load_world(self):
        skel_file = "skel/walker3d_waist.skel"
        world = pydart.World(SIMULATION_RATE, skel_file)
        skel = world.skeletons[-1]
        skel.set_self_collision_check(True)
        return world

    def consts(self):
        return consts_3D

def test_pd_control():
    from pd_control import PDController
    env = Simple3DEnv(PDController)
    env.reset()
    d = env.world.skeletons[1]
    q = d.q.copy()
    # Set some weird target pose
    q[6] = -1
    q[9] = np.pi/2
    q[12] = -np.pi * 0.75
    env.controller.target_q = q
    env.render()
    import time
    # Otherwise somehow it achieves the pose before the GUI launches
    time.sleep(0.1)
    for i in range(int(2 / SIMULATION_RATE)):
        env.step(60)

def test_simbicon():
    from simbicon_3D import Simbicon3D
    env = Simple3DEnv(Simbicon3D)
    env.reset(random=0.0)
    #env.sdf_loader.put_grounds([[0,0]], runway_length=20)
    for i in range(8):
        t = 0.3 + 0.4*i# + np.random.uniform(low=-0.2, high=0.2)
        env.simulate([t,0], render=1, put_dots=True)

if __name__ == '__main__':
    test_simbicon()

