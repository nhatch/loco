
from pd_control import PDController, KP_GAIN, KD_GAIN
import numpy as np
from IPython import embed
from inverse_kinematics import InverseKinematics

# The maximum time it should take to get (e.g.) the right foot off the ground
# after the left-foot heel strike.
LIFTOFF_DURATION = 0.3

class FSMState:
    def __init__(self, params):
        self.ik_gain               = params[0]
        self.position_balance_gain = params[1]
        self.velocity_balance_gain = params[2]
        self.torso_world           = params[3]
        self.swing_hip_world       = params[4]
        self.swing_knee_relative   = params[5]
        self.swing_ankle_relative  = params[6]
        self.stance_knee_relative  = params[7]
        self.stance_ankle_relative = params[8]
        # TODO avoid this redundancy. It already caused a bug that cost me two hours.
        self.raw_params = params

UP = 'UP'
DOWN = 'DOWN'

# Taken from Table 1 of https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf
# Then modified for the new parameters format.
walk = {
  UP: FSMState([0.14, 0.0, 0.2, 0.0, 0, 0, 0, 0, 0]),
  DOWN: FSMState([0.14, 0, 0, 0.0, 0, 0, 0.2, -0.1, 0.2])
  }

SIMBICON_ACTION_SIZE = 18

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
        self.target_x = 0
        self.stance_heel = 0
        self.direction = UP

    def set_gait_raw(self, target_x, raw_gait=None):
        up = walk[UP].raw_params.copy()
        down = walk[DOWN].raw_params.copy()
        if raw_gait is not None:
            up += raw_gait[0:9]
            down += raw_gait[9:18]
        gait = {UP: FSMState(up), DOWN: FSMState(down)}
        self.set_gait(gait)

        # All of these adjustments are just rough linear estimates from
        # fiddling around manually.
        self.target_x = target_x
        step_dist_diff = self.target_x - self.stance_heel - 0.4
        gait[UP].stance_knee_relative  += -0.05 - step_dist_diff * 0.2
        gait[UP].stance_ankle_relative += 0.2 + step_dist_diff * 0.4
        gait[UP].swing_hip_world       += 0.4 + step_dist_diff * 0.4
        gait[UP].swing_knee_relative   += -1.1 - step_dist_diff * 0.8

    def set_gait(self, gait):
        self.FSM = gait

    def state(self):
        return self.FSM[self.direction]

    def time(self):
        return self.env.world.time()

    def state_complete(self, left, right):
        swing_heel = self.ik.forward_kine(self.swing_idx)
        contacts_to_check = right if self.swing_idx == 3 else left
        if self.direction == UP:
            duration = self.time() - self.state_started
            early_strike = (duration >= LIFTOFF_DURATION) and (len(contacts_to_check) > 0)
            if early_strike:
                print("Early strike!")
            target_diff = self.state().ik_gain * self.skel.dq[0]
            # Start DOWN once heel is close enough to target (or time expires)
            heel_close = self.target_x < swing_heel[0] + target_diff
            com_close = self.target_x < self.skel.q[0] + target_diff
            if (heel_close and com_close) or early_strike:
                return True, None
        else:
            # Start UP once swing foot makes contact with the ground.
            # TODO there may be multiple contacts; which should we use?
            for contact in contacts_to_check:
                # TODO ideally put this in change_state (currently a side effect)
                self.contact_x = contact.p[0]
                prev_stance_heel = self.stance_heel
                self.stance_heel = swing_heel[0]
                return True, self.stance_heel - prev_stance_heel
        return False, None

    def change_state(self):
        self.state_started = self.time()
        if self.direction == UP:
            self.direction = DOWN
            self.calc_down_ik()
        else:
            self.direction = UP
            self.swing_idx, self.stance_idx = self.stance_idx, self.swing_idx
            res = "{:.2f} ({:+.2f})".format(self.stance_heel, self.stance_heel - self.target_x)
            return "{:.3f}: Ended step at {}".format(self.time(), res)

    def calc_down_ik(self):
        # Upon starting the DOWN part of the step, choose target swing leg angles
        # based on the location on the ground at target_x.
        tx = self.target_x - self.state().ik_gain * self.skel.dq[0]
        ty = -0.1 # TODO should we also adjust this based on vertical velocity?
        down, forward = self.ik.transform_frame(tx, ty)
        relative_hip, knee = self.ik.inv_kine(down, forward)
        self.FSM[DOWN].swing_hip_world = relative_hip + self.skel.q[2]
        self.FSM[DOWN].swing_knee_relative = knee

    def compute(self):
        state = self.state()
        q = np.zeros(9)
        q[self.stance_idx+1] = state.stance_knee_relative
        q[self.stance_idx+2] = state.stance_ankle_relative

        cd = state.position_balance_gain
        cv = state.velocity_balance_gain
        v = self.skel.dq[0]
        d = self.skel.q[0] - self.contact_x
        balance_feedback = cd*d + cv*v

        target_swing_angle = state.swing_hip_world + balance_feedback
        target_swing_knee = state.swing_knee_relative
        q[self.swing_idx+1] = target_swing_knee

        # The following line sets the swing ankle to be flat relative to the ground.
        q[self.swing_idx+2] = -(target_swing_angle + target_swing_knee)
        q[self.swing_idx+2] += state.swing_ankle_relative

        torso_actual = self.skel.q[2]
        q[self.swing_idx] = target_swing_angle - torso_actual

        self.target_q = q
        # We briefly increase Kd (mechanical impedance?) for the stance knee
        # in order to prevent the robot from stepping so hard that it bounces.
        fix_Kd = self.direction == UP and self.time() - self.state_started < 0.1
        if fix_Kd:
            self.Kd[self.stance_idx+1] *= 8
        control = super().compute()
        if fix_Kd:
            self.Kd[self.stance_idx+1] /= 8

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
    env.put_grounds([], runway_length=20)
    for i in range(10):
        # TODO: for very small target steps (e.g. 10 cm), the velocity is so small that
        # the robot can get stuck in the UP state, balancing on one leg.
        t = 0.3 + 0.8*i# + np.random.uniform(low=-0.2, high=0.2)
        env.simulate(t, render=1, put_dots=True)

