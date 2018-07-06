
from pd_control import PDController, KP_GAIN, KD_GAIN
import numpy as np
from IPython import embed
from inverse_kinematics import InverseKinematics

SIMBICON_ACTION_SIZE = 17
# This is assuming that the downstroke will take about 0.1s.
# If we're going 1 m/s, we should adjust our target_x back by 1 m/s * 0.1 s = 0.1 m.
IK_GAIN = 0.14

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
        self.raw_params = params

UP = 'UP'
DOWN = 'DOWN'

# Taken from Table 1 of https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf
walk = {
  UP: FSMState([0.3, 0.0, 0.2, 0.0, 0.4, -1.1, 0.2, -0.05, 0.2]),
  DOWN: FSMState([None, 2.2, 0.0, 0.0, -0.7, -0.05, 0.2, -0.1, 0.2])
  }

# Modified from the original because that didn't work on our model.
in_place_walk = {
  UP: FSMState([0.3, 0.0, 0.2, 0.0, 0.62, -1.1, 0.2, -0.15, 0.0]),
  DOWN: FSMState([None, 0.0, 0.0, 0.0, -0.1, -0.05, 0.2, -0.3, 0.1])
  }

_r = FSMState([0.35, 0.0, 0.2, 0.0, 0.8, -1.84, 0.2, -0.05, 0.27])
run = {
  DOWN: _r, UP: _r
  }

_fr = FSMState([0.21, 0.0, 0.2, -0.2, 1.08, -2.18, 0.2, -0.05, 0.27])
fast_run = {
  DOWN: _fr, UP: _fr
  }

GAIT_BIAS = np.concatenate((walk[UP].raw_params, walk[DOWN].raw_params[1:]))

class Simbicon(PDController):

    def __init__(self, skel, env):
        super().__init__(skel, env)
        self.reset()
        self.ik = InverseKinematics(env)

    def reset(self):
        self.state_started = self.time()
        self.swing_idx = 3
        self.stance_idx = 6
        self.contact_x = 0
        self.stance_heel = 0
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

    def time(self):
        return self.env.world.time()

    def state_complete(self, left, right):
        swing_heel = self.ik.forward_kine(self.swing_idx)
        contacts_to_check = right if self.swing_idx == 3 else left
        duration = self.time() - self.state_started
        if self.state().dwell_duration is not None:
            time_up = (duration >= self.state().dwell_duration) and (len(contacts_to_check) > 0)
            target_diff = IK_GAIN * self.skel.dq[0]
            if self.target_x < swing_heel[0] + target_diff or time_up:
                return True, None
            else:
                return False, None
        else:
            # This state should end when the swing foot makes contact with the ground.
            # TODO there may be multiple contacts; which should we use?
            for contact in contacts_to_check:
                # TODO ideally put this in change_state (currently a side effect)
                self.contact_x = contact.p[0]
                prev_stance_heel = self.stance_heel
                self.stance_heel = swing_heel[0]
                return True, self.stance_heel - prev_stance_heel
            return False, None

    def change_state(self):
        swing = "RIGHT" if self.stance_idx == 6 else "LEFT"
        suffix = " at {:.2f}".format(self.contact_x) if self.direction == DOWN else ""
        result = None
        if self.direction == DOWN:
            self.swing_idx, self.stance_idx = self.stance_idx, self.swing_idx
            err = self.stance_heel - self.target_x
            result = "{:.3f}: Ended state {} {}{} ({:+.2f})".format(self.time(), swing, self.direction, suffix, err)
            self.direction = UP
        else:
            # TODO skip this state if the swing foot is already in contact?
            self.direction = DOWN
            self.calc_down_ik()
        self.state_started = self.time()
        return result

    def calc_down_ik(self):
        if self.target_x is None:
            return
        # Upon starting the DOWN part of the step, choose target swing leg angles
        # based on the location on the ground at target_x.


        ty = -0.1 # TODO should we also adjust this based on vertical velocity?
        tx = self.target_x - IK_GAIN * self.skel.dq[0]
        down, forward = self.ik.transform_frame(tx, ty)
        relative_hip, knee = self.ik.inv_kine(down, forward)
        self.FSM[DOWN].swing_hip_world = relative_hip + self.skel.q[2]
        self.FSM[DOWN].swing_knee_relative = knee

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
        if self.direction == DOWN:
            balance_feedback = 0.0
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
    env = TwoStepEnv(Simbicon)
    env.controller.set_gait(walk)
    #env.seed(133712)
    #env.seed(42)
    env.reset(random=0.0)
    up = env.controller.FSM[UP]
    down = env.controller.FSM[DOWN]
    for i in range(12):
        t = 0.3 + 0.4*i
        env.simulate(render=1.0, target_x=t)

