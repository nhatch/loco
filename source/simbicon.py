
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
        self.raw_params = params

UP = 'UP'
DOWN = 'DOWN'

# Taken from Table 1 of https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf
# Then modified for the new parameters format.
walk = {
  UP: FSMState([0.14, 0.0, 0.2, 0.0, 0.4, -1.1, 0.2, -0.05, 0.2]),
  DOWN: FSMState([0.14, 0, 0, 0.0, 0, 0, 0.2, -0.1, 0.2])
  }

SIMBICON_ACTION_SIZE = 14

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

    def set_gait_raw(self, raw_gait):
        up = raw_gait[0:9]
        up = walk[UP].raw_params.copy()
        # Skip the parameters that are not configurable
        up += raw_gait[0:9]
        down = walk[DOWN].raw_params.copy()
        down[0:1] += raw_gait[9:10]
        down[3:4] += raw_gait[10:11]
        down[6:9] += raw_gait[11:14]
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
        if self.direction == UP:
            duration = self.time() - self.state_started
            early_strike = (duration >= LIFTOFF_DURATION) and (len(contacts_to_check) > 0)
            if early_strike:
                print("Early strike!")
            target_diff = self.state().ik_gain * self.skel.dq[0]
            # Start DOWN once heel is close enough to target (or time expires)
            if self.target_x < swing_heel[0] + target_diff or early_strike:
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

    def set_target(self, target):
        # TODO: all of these gain adjustments are just rough linear estimates from
        # fiddling around manually. In theory, learning the inverse dynamics with
        # a linear model should do all this work automatically.
        self.target_x = target
        step_dist_diff = self.target_x - self.stance_heel - 0.4
        self.FSM[UP].stance_knee_relative = -0.05 - step_dist_diff * 0.6
        self.FSM[UP].stance_ankle_relative = 0.2 + step_dist_diff * 0.6
        self.FSM[UP].swing_hip_world = 0.4 + step_dist_diff * 0.1

        self.FSM[UP].ik_gain = 0.14 # good for target length 0.2-0.4
        #self.FSM[UP].ik_gain = 0.09 # good for target length 0.5
        #self.FSM[UP].ik_gain = 0.04 # good for target length 0.6
        #self.FSM[UP].swing_knee_relative = -1.1

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
        q[self.swing_idx+1] = state.swing_knee_relative
        # TODO maybe set swing ankle target in world space?
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
        # We increase Kd (mechanical impedance?) for the stance knee
        # in order to prevent the robot from stepping so hard that it bounces.
        self.Kd[self.stance_idx+1] *= 2
        control = super().compute()
        self.Kd[self.stance_idx+1] /= 2

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
    for i in range(20):
        # TODO: for very small target steps (e.g. 10 cm), the velocity is so small that
        # the robot can get stuck in the UP state, balancing on one leg.
        t = 0.3 + 0.4*i + np.random.uniform(low=-0.2, high=0.2)
        env.simulate(t, render=1, show_dots=True)

