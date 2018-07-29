import pydart2 as pydart
from IPython import embed
import numpy as np

from pydart2.gui.trackball import Trackball, _q_add, _q_rotmatrix
from OpenGL.GL import GLfloat

from stepping_stones_env import SteppingStonesEnv
import consts_3D
from state_3D import State3D

SIMULATION_RATE = 1.0 / 2000.0
THETA = -np.pi/6
PHI = np.pi / 2

class Simple3DEnv(SteppingStonesEnv):
    def update_viewer(self, com):
        x0 = com[0]
        # Transform the offset -[x0, 0, 0] into the camera coordinate frame.
        # First, rotate by PHI around the Y axis.
        x1 = x0 * np.cos(self.phi)
        z1 = x0 * np.sin(self.phi)
        # Then, rotate by THETA around the new X axis.
        x2 = x1
        y2 = z1 * np.sin(self.theta)
        z2 = z1 * np.cos(self.theta)
        # Move z_2 back by 5 so the camera has some distance from the agent.
        trans = [-x2, -y2, -(z2 + 5)]
        self.viewer.scene.tb.trans[0:3] = trans

    def set_rot(self, tb):
        # The default Trackball parameters rotate phi around the z axis
        # rather than the y axis. I modified this code from the
        # Trackball.set_orientation method.
        xrot = np.array([np.sin(self.theta/2), 0, 0, np.cos(self.theta/2)])
        yrot = np.array([0, np.sin(self.phi/2), 0, np.cos(self.phi/2)])
        rot = _q_add(xrot, yrot)
        tb._rotation = rot
        m = _q_rotmatrix(rot)
        tb._matrix = (GLfloat*len(m))(*m)

    def init_camera(self):
        self.theta = THETA
        self.phi = PHI
        tb = Trackball()
        self.set_rot(tb)
        # Overwrite the drag_to method so we only change phi, not theta.
        def drag_to(x,y,dx,dy):
            self.phi += -dx/80
            self.set_rot(self.viewer.scene.tb)
        tb.drag_to = drag_to
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

