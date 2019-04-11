from IPython import embed
import numpy as np

from simbicon_3D import Simbicon3D
import consts_darwin

from learn_llc import EMBED_B5
import time

SAVED_TRAJ = np.loadtxt('data/b5_controls.txt')

class RealDarwinEnv:
    def __init__(self, controller_class=Simbicon3D):
        self.robot_skeleton = self.load_robot()
        self.t0 = time.time()
        self.controller = controller_class(self)
        self.is_3D = True

    def time(self):
        return time.time() - self.t0

    def consts(self):
        return consts_darwin

    def load_robot(self):
        self.robot_skeleton = None

    def control_tick(self, i):
        c = self.consts()
        if self.controller.swing_contact(None, None): # TODO rename: not contact, just timeout
            status_string = self.controller.change_stance(np.zeros(3))
            self.controller.set_gait_raw(raw_gait=EMBED_B5, target_heading=None, target=None)
            print(status_string)
        #q, dq = self.talk_to_sensors()
        #target_q = c.raw_dofs(self.controller.compute_target_q(q, dq))
        target_q = SAVED_TRAJ[i]
        self.send_to_robot(target_q)

    def talk_to_sensors(self):
        c = self.consts()
        q = np.zeros(c.Q_DIM)
        dq = np.zeros(c.Q_DIM)
        return q, dq

    def send_to_robot(self, target_q):
        return

if __name__ == '__main__':
    env = RealDarwinEnv()
    i = 0
    freq = 40 # Hz
    duration = 5 # seconds
    env.controller.set_gait_raw(raw_gait=EMBED_B5, target_heading=None, target=None)
    while i < freq*duration:
        env.control_tick(i)
        i += 1
        time_left = i/freq - env.time()
        if time_left > 0:
            time.sleep(time_left)
