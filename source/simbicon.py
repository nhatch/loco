
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

    def reset(self, state=np.zeros(8)):
        self.step_started = self.time()
        self.swing_idx = 3
        self.stance_idx = 6
        self.contact = state[0:2]
        self.stance_heel = state[2:4]
        self.target = state[4:6]
        # We track prev_target so that, in principle, we know all of the possible
        # places where the agent is in contact with the environment. However, the
        # algorithm does not use this (yet) except when resetting the environment
        # to a previously visited state.
        self.prev_target = state[6:8]
        self.direction = UP

    def set_gait_raw(self, target, raw_gait=None):
        up = walk[UP].raw_params.copy()
        down = walk[DOWN].raw_params.copy()
        if raw_gait is not None:
            up += raw_gait[0:9]
            down += raw_gait[9:18]
        gait = {UP: FSMState(up), DOWN: FSMState(down)}
        self.set_gait(gait)

        self.target, self.prev_target = target, self.target

        # Calculate a normal vector for the line between the starting location
        # of the swing heel and the target.
        # This is used when detecting crashes and evaluating progress of
        # the learning algorithm.
        swing_heel = self.ik.forward_kine(self.swing_idx)[:2]
        self.starting_swing_heel = swing_heel
        d = self.target - self.starting_swing_heel
        d /= np.linalg.norm(d)
        self.unit_normal = np.array([-d[1], d[0]])

        # All of these adjustments are just rough linear estimates from
        # fiddling around manually.
        step_dist_diff = self.target[0] - self.stance_heel[0] - 0.4
        gait[UP].stance_knee_relative  += -0.05 - step_dist_diff * 0.2
        gait[UP].stance_ankle_relative += 0.2 + step_dist_diff * 0.4
        gait[UP].swing_hip_world       += 0.4 + step_dist_diff * 0.4
        gait[UP].swing_knee_relative   += -1.1 - step_dist_diff * 0.8

    def set_gait(self, gait):
        self.FSM = gait

    def state(self):
        return np.concatenate((self.contact, self.stance_heel, self.target, self.prev_target))

    def FSMstate(self):
        return self.FSM[self.direction]

    def time(self):
        return self.env.world.time()

    def crashed(self, swing_heel):
        tol = -0.02
        lower = min(self.starting_swing_heel[1], self.target[1])
        upper = max(self.starting_swing_heel[1], self.target[1])
        below_lower = swing_heel[1] - lower < tol
        below_upper = swing_heel[1] - upper < tol
        # Calculate the distance of the swing heel from the line between the
        # start and end targets.
        below_line = np.dot(swing_heel - self.starting_swing_heel, self.unit_normal) < tol
        if below_lower or (below_line and below_upper):
            return False, True

    # Returns True if either the swing foot has made contact, or if it seems impossible
    # that the swing foot will make contact in the future.
    def step_complete(self, contacts, swing_heel):
        if self.crashed(swing_heel):
            return True
        elif self.direction == UP:
            self.maybe_start_down_phase(contacts, swing_heel)
            return False
        elif len(contacts) > 0:
            return True
        else:
            return False

    def maybe_start_down_phase(self, contacts, swing_heel):
        duration = self.time() - self.step_started
        early_strike = (duration >= LIFTOFF_DURATION) and (len(contacts) > 0)
        if early_strike:
            print("Early strike!")
        target_diff = self.FSMstate().ik_gain * self.skel.dq[0]
        heel_close = self.target[0] < swing_heel[0] + target_diff
        com_close = self.target[0] < self.skel.q[0] + target_diff
        if (heel_close and com_close) or early_strike:
            # Start the DOWN phase
            self.direction = DOWN
            self.calc_down_ik()

    def change_stance(self, contacts, swing_heel):
        self.step_started = self.time()
        if len(contacts) > 0:
            # TODO which contact should we use?
            self.contact = contacts[0].p[0:2]
        else:
            # The swing foot missed the target
            self.contact = -np.inf * np.ones(2)
        self.stance_heel = swing_heel
        self.direction = UP
        self.swing_idx, self.stance_idx = self.stance_idx, self.swing_idx
        hx, hy = self.stance_heel
        dx, dy = self.stance_heel - self.target
        res = "{:.2f}, {:.2f} ({:+.2f}, {:+.2f})".format(hx, hy, dx, dy)
        return "{:.3f}: Ended step at {}".format(self.time(), res)

    def calc_down_ik(self):
        # Upon starting the DOWN part of the step, choose target swing leg angles
        # based on the location on the ground at target.
        tx = self.target[0] - self.FSMstate().ik_gain * self.skel.dq[0]
        ty = -0.1 # TODO should we also adjust this based on vertical velocity?
        down, forward = self.ik.transform_frame(tx, ty)
        relative_hip, knee = self.ik.inv_kine(down, forward)
        self.FSM[DOWN].swing_hip_world = relative_hip + self.skel.q[2]
        self.FSM[DOWN].swing_knee_relative = knee

    def compute(self):
        state = self.FSMstate()
        q = np.zeros(9)
        q[self.stance_idx+1] = state.stance_knee_relative
        q[self.stance_idx+2] = state.stance_ankle_relative

        cd = state.position_balance_gain
        cv = state.velocity_balance_gain
        v = self.skel.dq[0]
        d = self.skel.q[0] - self.contact[0]
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
        fix_Kd = self.direction == UP and self.time() - self.step_started < 0.1
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
    #env.seed(133712)
    #env.seed(42)
    env.reset(random=0.0)
    env.sdf_loader.put_grounds([[0,0]], runway_length=20)
    for i in range(10):
        # TODO: for very small target steps (e.g. 10 cm), the velocity is so small that
        # the robot can get stuck in the UP state, balancing on one leg.
        t = 0.3 + 0.8*i# + np.random.uniform(low=-0.2, high=0.2)
        env.simulate([t,0], render=1, put_dots=True)

