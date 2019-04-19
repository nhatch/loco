from IPython import embed
import numpy as np

from simbicon_3D import Simbicon3D
import consts_darwin
from hardware.darwin_interface import DarwinInterface

from learn_llc import EMBED_B5
import time

SAVED_TRAJ = np.loadtxt('data/b5_controls.txt')
CONTROL_FREQUENCY = 40. # Hz

class RealDarwinEnv:
    def __init__(self, controller_class=Simbicon3D):
        self.robot_skeleton = self.load_robot()
        self.is_3D = True
        self.t0 = time.time()
        self.prev_time = self.t0
        self.prev_control_time = self.t0
        self.controller = controller_class(self)
        self.control_tick = 0
        self.imu_tick = 0

    def time(self):
        return time.time() - self.t0

    def consts(self):
        return consts_darwin

    def load_robot(self):
        init_state = np.zeros(self.consts().Q_DIM_RAW)
        self.robot = DarwinInterface(init_state)

    def tick(self):
        self.imu_tick += 1
        t = self.time()
        self.robot.integrate_imu(self.prev_time, t)
        if self.time() >= self.control_tick / CONTROL_FREQUENCY:
            self.update_control(t)
        self.prev_time = t

    def update_control(self, t):
        c = self.consts()
        if self.controller.swing_contact(None, None): # TODO rename: not contact, just timeout
            status_string = self.controller.change_stance(np.zeros(3))
            self.controller.set_gait_raw(raw_gait=EMBED_B5, target_heading=None, target=None)
            print(status_string)
        q, dq = self.robot.read(self.prev_control_time, t)
        target_q = c.raw_dofs(self.controller.compute_target_q(q, dq))
        target_q = SAVED_TRAJ[self.control_tick]
        self.robot.write(target_q)
        self.prev_control_time = t
        self.control_tick += 1

if __name__ == '__main__':
    env = RealDarwinEnv()
    duration = 5 # seconds
    env.controller.set_gait_raw(raw_gait=EMBED_B5, target_heading=None, target=None)
    while env.time() < duration:
        env.tick()
    print("Control frequency: ", env.control_tick//duration)
    print("IMU read frequency:", env.imu_tick//duration)