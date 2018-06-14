
from walker import PDController, KP_GAIN, KD_GAIN
import numpy as np

class FSMState:
    def __init__(self, params):
        self.dwell_duration        = params[0]
        self.position_balance_gain = params[1]
        self.velocity_balance_gain = params[2]
        self.torso_world           = params[3]
        self.swing_hip_world       = params[4]
        self.swing_knee_relative   = params[5]
        self.swing_ankle_relative  = params[6]
        self.stance_knee_relative  = params[7]
        self.stance_ankle_relative = params[8]

# Taken from Table 1 of https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf
FSM = {"up": FSMState([0.3, 0.0, 0.2, 0.0, 0.4, -1.1, 0.2, -0.05, 0.2]),
        "down": FSMState([None, 2.2, 0.0, 0.0, -0.7, -0.05, 0.2, -0.1, 0.2])}

class Simbicon(PDController):

    def __init__(self, skel, world):
        super().__init__(skel, world)
        self.state = FSM['up']
        self.state_started = self.world.time()
        self.swing_idx = 3
        self.stance_idx = 6
        self.contact_x = 0

    def state_complete(self, left, right):
        if self.state.dwell_duration is not None:
            if self.world.time() - self.state_started >= self.state.dwell_duration:
                return True
        else:
            if right is not None and self.swing_idx == 3:
                self.contact_x = right.p[0] # TODO ideally put this in change_state (currently a side effect)
                return True
            elif left is not None and self.swing_idx == 6:
                self.contact_x = left.p[0]
                return True

    def change_state(self):
        if self.state == FSM['down']:
            self.swing_idx, self.stance_idx = self.stance_idx, self.swing_idx
            self.state = FSM['up']
        else:
            # TODO skip this state if the swing foot is already in contact?
            self.state = FSM['down']
        self.state_started = self.world.time()

    def compute(self):
        q = self.target_q.copy()
        q[self.swing_idx+1] = self.state.swing_knee_relative
        q[self.swing_idx+2] = self.state.swing_ankle_relative
        q[self.stance_idx+1] = self.state.stance_knee_relative
        q[self.stance_idx+2] = self.state.stance_ankle_relative
        self.target_q = q

        cd = self.state.position_balance_gain
        cv = self.state.velocity_balance_gain
        v = self.skel.dq[0]
        d = self.skel.q[0] - self.contact_x
        balance_feedback = cd*d + cv*v

        torso_actual = self.skel.q[2]
        self.target_q[self.swing_idx] = self.state.swing_hip_world - torso_actual + balance_feedback

        control = super().compute()
        torso_torque = - KP_GAIN * (torso_actual - self.state.torso_world) - KD_GAIN * self.skel.dq[2]
        control[self.stance_idx] = -torso_torque - control[self.swing_idx]
        return control

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon)
    from random_search import Whitener
    w = Whitener(env, False)
    zero_ctrl = lambda _: np.zeros(6)
    w.run_trajectory(zero_ctrl, 0, True, False)
