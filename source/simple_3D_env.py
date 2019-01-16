import pydart2 as pydart
from IPython import embed
import numpy as np

from pydart2.gui.trackball import Trackball, _q_add, _q_rotmatrix
from OpenGL.GL import GLfloat

from stepping_stones_env import SteppingStonesEnv
import consts_armless as consts

SIMULATION_RATE = 1.0 / 2000.0
THETA = -np.pi/12
PHI = np.pi / 1.5
ZOOM = 5.0

class Simple3DEnv(SteppingStonesEnv):
    def update_viewer(self, com):
        x0, _, z0 = self.track_point or com
        # Transform the offset -[x0, 0, z0] into the camera coordinate frame.
        # First, rotate by PHI around the Y axis.
        x1 = x0 * np.cos(self.phi) - z0 * np.sin(self.phi)
        z1 = x0 * np.sin(self.phi) + z0 * np.cos(self.phi)
        # Then, rotate by THETA around the new X axis.
        x2 = x1
        y2 = z1 * np.sin(self.theta)
        z2 = z1 * np.cos(self.theta)
        # Move z_2 back by `zoom` so the camera has some distance from the agent.
        trans = [-x2, -y2, -(z2+self.zoom)]
        self.viewer.scene.tb.trans[0:3] = trans

    def set_rot_manual(self, phi):
        self.phi = phi
        self.set_rot(self.viewer.scene.tb)

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
        self.zoom = ZOOM
        self.track_point = None
        tb = Trackball()
        self.set_rot(tb)
        # Overwrite the drag_to method so we only change phi, not theta.
        def drag_to(x,y,dx,dy):
            self.phi += -dx/80
            self.set_rot(self.viewer.scene.tb)
        def zoom_to(dx, dy):
            self.zoom -= dy/20
        tb.drag_to = drag_to
        tb.zoom_to = zoom_to
        return tb

    def step(self, frames_per_second=60):
        self.world.step()
        framerate = int(1 / SIMULATION_RATE / frames_per_second)
        if self.world.frame % framerate == 0:
            #print(self.world.time())
            self._render()

    def load_world(self):
        skel_file = consts.skel_file
        world = pydart.World(SIMULATION_RATE, skel_file)
        skel = world.skeletons[1]
        self.doppelganger = None
        if len(world.skeletons) == 3:
            self.doppelganger = world.skeletons[2]
            assert(self.doppelganger.name == "doppelganger")
        skel.set_self_collision_check(True)
        # The kima_human_box model actually has preset damping coeffs.
        #for dof in skel.dofs[6:]:
        #    dof.set_damping_coefficient(0.2)
        return world

    def consts(self):
        return consts

def test_pd_control():
    from pd_control import PDController
    env = Simple3DEnv(PDController)
    env.reset()
    env.sdf_loader.put_grounds([[-5,-0.9,0]])
    q, _ = env.get_x()
    env.render()
    # Set some weird target pose
    c = env.consts()
    q[c.RIGHT_IDX + c.HIP_PITCH] = 1
    q[c.RIGHT_IDX + c.KNEE] = -np.pi/2
    q[c.LEFT_IDX + c.HIP_PITCH] = -np.pi * 0.75
    env.controller.target_q = q
    env.render()
    import time
    # Otherwise somehow it achieves the pose before the GUI launches
    time.sleep(0.1)
    for i in range(int(2 / SIMULATION_RATE)):
        env.step(60)

def test_gimbal_lock(joint):
    from pd_control import PDController
    from sdf_loader import RED, GREEN, BLUE
    env = Simple3DEnv(PDController)
    env.track_point = [0,0,0]
    env.sdf_loader.put_dot([1.5,0,0], "positive_x", color=RED)
    env.sdf_loader.put_dot([0,1.5,0], "positive_y", color=GREEN)
    env.sdf_loader.put_dot([0,0,1.5], "positive_z", color=BLUE)
    q = env.current_observation()
    def t(a,b,c):
        q.raw_state[joint] = [a,b,c]; env.reset(q); env.render()
    import time
    t(0,0,0)
    time.sleep(1)
    t(0,0,0.5)
    time.sleep(1)
    t(0.5,0.0,0) # Should look different from previous (i.e. no gimbal lock)
    time.sleep(1)
    t(0.5,1,0) # Should just rotate around the Z axis
    time.sleep(1)
    t(0.5, np.pi/2, 0) # Same as above (heading is now perpendicular to original)
    time.sleep(1)
    # NOTE: for the hip joint there is actually gimbal lock here, but that's ok
    t(0.25, np.pi/2, 0.25) # Should look different from above (i.e. no gimbal lock)
    time.sleep(1)

if __name__ == "__main__":
    test_pd_control()
    ROOT = [3,4,5]
    RIGHT_HIP = [6,7,8]
    LEFT_HIP = [12,13,14]
    test_gimbal_lock(LEFT_HIP)
