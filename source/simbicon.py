
from pd_control import PDController, KP_GAIN, KD_GAIN
import numpy as np
from IPython import embed

SIMBICON_ACTION_SIZE = 17

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

UP = 'up'
DOWN = 'down'

# Taken from Table 1 of https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf
walk = {
  UP: FSMState([0.3, 0.0, 0.2, 0.0, 0.4, -1.1, 0.2, -0.05, 0.2]),
  DOWN: FSMState([None, 2.2, 0.0, 0.0, -0.7, -0.05, 0.2, -0.1, 0.2])
  }

_r = FSMState([0.35, 0.0, 0.2, 0.0, 0.8, -1.84, 0.2, -0.05, 0.27])
run = {
  DOWN: _r, UP: _r
  }

_fr = FSMState([0.21, 0.0, 0.2, -0.2, 1.08, -2.18, 0.2, -0.05, 0.27])
fast_run = {
  DOWN: _fr, UP: _fr
  }

REAL_TIME = 1.2
RENDER_FACTOR = 1.2
GAIT = fast_run

# Walking gait, so that a zero controller still behaves reasonably
GAIT_BIAS = np.array([
    0.3, 0.0, 0.2, 0.0, 0.4, -1.1, 0.2, -0.05, 0.2,
    2.2, 0.0, 0.0, -0.7, -0.05, 0.2, -0.1, 0.2])

class Simbicon(PDController):

    def __init__(self, skel, world):
        super().__init__(skel, world)
        self.reset()

    def reset(self):
        self.state_started = self.world.time()
        self.swing_idx = 3
        self.stance_idx = 6
        self.contact_x = 0
        self.direction = UP

    def set_gait_raw(self, raw_gait_centered):
        raw_gait = raw_gait_centered + GAIT_BIAS
        up = raw_gait[0:9]
        down = np.concatenate(([None], raw_gait[9:SIMBICON_ACTION_SIZE]))
        gait = {UP: FSMState(up), DOWN: FSMState(down)}
        self.set_gait(gait)

    def set_gait(self, gait):
        self.FSM = gait

    def state(self):
        return self.FSM[self.direction]

    def state_complete(self, left, right):
        if self.state().dwell_duration is not None:
            if self.world.time() - self.state_started >= self.state().dwell_duration:
                return True, None
            else:
                return False, None
        else:
            prev_x = self.contact_x
            if right is not None and self.swing_idx == 3:
                self.contact_x = right.p[0] # TODO ideally put this in change_state (currently a side effect)
                return True, self.contact_x - prev_x
            elif left is not None and self.swing_idx == 6:
                self.contact_x = left.p[0]
                return True, self.contact_x - prev_x
            else:
                return False, None

    def change_state(self):
        swing = "RIGHT" if self.stance_idx == 6 else "LEFT"
        suffix = " at {:.3f}".format(self.contact_x) if self.direction == DOWN else ""
        print("Ended state {} {}{}".format(swing, self.direction, suffix))
        if self.direction == DOWN:
            self.swing_idx, self.stance_idx = self.stance_idx, self.swing_idx
            self.direction = UP
        else:
            # TODO skip this state if the swing foot is already in contact?
            self.direction = DOWN
        self.state_started = self.world.time()

    def compute(self):
        # TODO the walking controller can start from rest,
        # but not from (rest + small perturbation). (It does the splits and falls over.)
        state = self.state()
        q = np.zeros(9)
        q[self.swing_idx+1] = state.swing_knee_relative
        q[self.swing_idx+2] = state.swing_ankle_relative
        q[self.stance_idx+1] = state.stance_knee_relative
        q[self.stance_idx+2] = state.stance_ankle_relative

        cd = state.position_balance_gain
        cv = state.velocity_balance_gain
        v = self.skel.dq[0]
        d = self.skel.q[0] - self.contact_x
        balance_feedback = cd*d + cv*v
        target_swing_angle = state.swing_hip_world + balance_feedback

        torso_actual = self.skel.q[2]
        q[self.swing_idx] = target_swing_angle - torso_actual

        self.target_q = q
        control = super().compute()
        torso_torque = - KP_GAIN * (torso_actual - state.torso_world) - KD_GAIN * self.skel.dq[2]
        control[self.stance_idx] = -torso_torque - control[self.swing_idx]
        return control

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon, render_factor = 2*RENDER_FACTOR)
    env.controller.set_gait(walk)
    from random_search import Whitener
    w = Whitener(env, False)
    zero_ctrl = lambda _: np.zeros(env.action_space.shape[0])
    env.reset()
    w.run_trajectory(zero_ctrl, 0, True, False)
