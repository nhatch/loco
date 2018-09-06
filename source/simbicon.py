
from pd_control import PDController
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
        self.position_balance_gain_lat = params[9]
        self.velocity_balance_gain_lat = params[10]
        # TODO avoid this redundancy. It already caused a bug that cost me two hours.
        self.raw_params = params

UP = 'UP'
DOWN = 'DOWN'

SIMBICON_ACTION_SIZE = 22

class Simbicon(PDController):

    def __init__(self, skel, env):
        self.ik = InverseKinematics(env)
        super().__init__(skel, env)

    def reset(self, state=None):
        c = self.env.consts()
        self.step_started = self.time()
        self.swing_idx = c.RIGHT_IDX
        self.stance_idx = c.LEFT_IDX

        if state is None:
            state = np.zeros(9)
            stance_heel = self.ik.forward_kine(self.stance_idx)
            state[0:3] = stance_heel

        self.stance_heel = state[0:3]
        self.target = state[3:6]
        # We track prev_target so that, in principle, we know all of the possible
        # places where the agent is in contact with the environment. However, the
        # algorithm does not use this (yet) except when resetting the environment
        # to a previously visited state.
        self.prev_target = state[6:9]
        self.direction = UP

    def standardize_stance(self, state):
        c = self.env.consts()
        # Ensure the stance leg is the right leg.
        if self.swing_idx == c.RIGHT_IDX:
            return state
        # We need to flip the left and right leg.
        D = c.LEG_DOF
        R = c.RIGHT_IDX
        L = c.LEFT_IDX
        for base in [0, c.Q_DIM]:
            right = state[base+R : base+R+D].copy()
            left  = state[base+L : base+L+D].copy()
            state[base+R : base+R+D] = left
            state[base+L : base+L+D] = right
        return state

    def base_gait(self):
        # Taken from Table 1 of https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf
        # Then modified for the new parameters format.
        gait = ([0.14, 0, 0.2, 0.0, 0.4, -1.1,   0, -0.05, 0.2, 0.5, 0.2],
                [0.14, 0,   0, 0.0,   0,    0, 0.2, -0.1,  0.2, 0.5, 0.2])
        # This is the gait that works best in 3D. It doesn't work well, though.
        # Uncomment this if you want to see what the gait looks like in 2D.
        # NOTE: You will need to stub out adjust_up_targets and maybe_start_down_phase
        # to match the implementations in simbicon_3D.py.
        # gait = ([0.14, 0.5, 0.2, -0.2,  0.5, -1.1,   0.6, -0.05,  0, 0.5, 0.2],
        #         [0.14, 0.5, 0.2, -0.2, -0.1, -0.05, 0.15, -0.1, 0.0, 0.5, 0.2])
        return gait

    def set_gait_raw(self, target, raw_gait=None):
        up, down = self.base_gait()
        if raw_gait is not None:
            up += raw_gait[:len(up)]
            down += raw_gait[len(up):]
        gait = {UP: FSMState(up), DOWN: FSMState(down)}
        self.set_gait(gait)

        self.target, self.prev_target = target, self.target

        # Calculate a normal vector for the line between the starting location
        # of the swing heel and the target.
        # This is used when detecting crashes and evaluating progress of
        # the learning algorithm.
        swing_heel = self.ik.forward_kine(self.swing_idx)
        self.starting_swing_heel = swing_heel
        d = self.target - self.starting_swing_heel
        d = np.array([-d[1], d[0], 0])
        self.unit_normal = d / np.linalg.norm(d)

        self.adjust_up_targets()

    def adjust_up_targets(self):
        # All of these adjustments are just rough linear estimates from
        # fiddling around manually.
        step_dist_diff = self.target[0] - self.stance_heel[0] - 0.4
        gait = self.FSM
        gait[UP].stance_knee_relative  += - step_dist_diff * 0.2
        gait[UP].stance_ankle_relative += + step_dist_diff * 0.4
        gait[UP].swing_hip_world       += + step_dist_diff * 0.4
        gait[UP].swing_knee_relative   += - step_dist_diff * 0.8

    def set_gait(self, gait):
        self.FSM = gait

    def state(self):
        return np.concatenate((self.stance_heel, self.target, self.prev_target))

    def FSMstate(self):
        return self.FSM[self.direction]

    def time(self):
        return self.env.world.time()

    def crashed(self, swing_heel):
        c = self.env.consts()
        for contact in self.env.world.collision_result.contacts:
            bodynode = contact.bodynode2 if contact.skel_id1 == 0 else contact.bodynode1
            if contact.skel_id1 == contact.skel_id2:
                # The robot crashed into itself
                print("SELF COLLISION")
                return True
            if not bodynode.id in c.ALLOWED_COLLISION_IDS:
                print("HIT THE GROUND")
                return True
        # For some reason, setting the tolerance smaller than .05 or so causes the controller
        # to learn very weird behaviors. TODO: why does this have such a large effect??
        # However, setting the tolerance too large (larger than .04 or so) makes certain crashes
        # go undetected. E.g.:
        #
        #   python inverse_dynamics.py     # Don't run any training iters or load the train set
        #   learn.evaluate(seed=39738)
        tol = -0.06
        lower = min(self.starting_swing_heel[1], self.target[1])
        upper = max(self.starting_swing_heel[1], self.target[1])
        below_lower = swing_heel[1] - lower < tol
        below_upper = swing_heel[1] - upper < tol
        # Calculate the distance of the swing heel from the line between the
        # start and end targets.
        below_line = np.dot(swing_heel - self.starting_swing_heel, self.unit_normal) < tol
        if below_lower or (below_line and below_upper):
            print("FELL OFF")
            return True

    # Returns True if either the swing foot has made contact, or if it seems impossible
    # that the swing foot will make contact in the future.
    def step_complete(self, contacts, swing_heel):
        if self.crashed(swing_heel):
            return True
        elif self.direction == UP:
            self.maybe_start_down_phase(contacts, swing_heel)
            return False
        elif len(contacts) > 0: # and self.direction == DOWN
            return True
        else:
            return False

    def maybe_start_down_phase(self, contacts, swing_heel):
        c = self.env.consts()
        duration = self.time() - self.step_started
        early_strike = (duration >= LIFTOFF_DURATION) and (len(contacts) > 0)
        if early_strike:
            print("Early strike!")
        q, dq = self.env.get_x()
        target_diff = self.FSMstate().ik_gain * dq[0]
        heel_close = self.target[0] < swing_heel[0] + target_diff
        com_close = self.target[0] < q[0] + target_diff
        if (heel_close and com_close) or early_strike:
            # Start the DOWN phase
            self.direction = DOWN
            self.calc_down_ik()

    def change_stance(self, contacts, swing_heel):
        self.step_started = self.time()
        self.stance_heel = swing_heel
        self.direction = UP
        self.swing_idx, self.stance_idx = self.stance_idx, self.swing_idx
        hx, hy, hz = self.stance_heel
        dx, dy, dz = self.stance_heel - self.target
        res = "{:.2f}, {:.2f} ({:+.2f}, {:+.2f})".format(hx, hy, dx, dy)
        return "{:.3f}: Ended step at {}".format(self.time(), res)

    def calc_down_ik(self):
        c = self.env.consts()
        q, dq = self.env.get_x()
        # Upon starting the DOWN part of the step, choose target swing leg angles
        # based on the location on the ground at target.
        tx = self.target[0] - self.FSMstate().ik_gain * dq[0]
        ty = self.target[1] - 0.1 # TODO should we adjust this based on vertical velocity?
        relative_hip, knee = self.ik.inv_kine([tx, ty])
        self.FSM[DOWN].swing_hip_world = relative_hip + q[c.PITCH_IDX]
        self.FSM[DOWN].swing_knee_relative = knee

    def compute_target_q(self, q, dq):
        c = self.env.consts()
        state = self.FSMstate()
        tq = np.zeros(c.Q_DIM)
        tq[self.stance_idx+c.KNEE_IDX] = state.stance_knee_relative
        tq[self.stance_idx+c.KNEE_IDX+1] = state.stance_ankle_relative

        cd = state.position_balance_gain
        cv = state.velocity_balance_gain
        v = dq[0]
        d = q[0] - self.stance_heel[0]
        balance_feedback = cd*d + cv*v

        target_swing_angle = state.swing_hip_world + balance_feedback
        target_swing_knee = state.swing_knee_relative
        tq[self.swing_idx+c.KNEE_IDX] = target_swing_knee

        # The following line sets the swing ankle to be flat relative to the ground.
        tq[self.swing_idx+c.KNEE_IDX+1] = -(target_swing_angle + target_swing_knee)
        tq[self.swing_idx+c.KNEE_IDX+1] += state.swing_ankle_relative

        torso_actual = q[c.PITCH_IDX]
        tq[self.swing_idx] = target_swing_angle - torso_actual
        return tq

    def compute(self):
        c = self.env.consts()
        q, dq = self.env.get_x()
        state = self.FSMstate()
        target_q = self.compute_target_q(q, dq)
        # We briefly increase Kd (mechanical impedance?) for the stance knee
        # in order to prevent the robot from stepping so hard that it bounces.
        fix_Kd = self.direction == UP and self.time() - self.step_started < 0.1
        if fix_Kd:
            self.Kd[self.stance_idx+c.KNEE_IDX] *= 8
        control = self.compute_transformed(target_q)
        if fix_Kd:
            self.Kd[self.stance_idx+c.KNEE_IDX] /= 8

        # Make modifications to control torso pitch
        torso_actual = q[c.PITCH_IDX]
        torso_speed = dq[c.PITCH_IDX]
        kp = self.Kp[self.stance_idx]
        kd = self.Kd[self.stance_idx]
        torso_torque = - kp * (torso_actual - state.torso_world) - kd * torso_speed
        control[self.stance_idx] = -torso_torque - control[self.swing_idx]

        # The following were hacks to attempt to make the 3D model maintain its torso
        # facing straight forward. They didn't work well but here's the code if you want.
        #torso_actual = q[c.ROLL_IDX]
        #torso_speed = dq[c.ROLL_IDX]
        #torso_torque = - c.KP_GAIN * (torso_actual - 0) - c.KD_GAIN * torso_speed
        #control[self.stance_idx+c.HIP_OFFSET_LAT] = -torso_torque# - control[self.swing_idx+c.HIP_OFFSET_LAT]
        #torso_actual = q[c.YAW_IDX]
        #torso_speed = dq[c.YAW_IDX]
        #torso_torque = - c.KP_GAIN * (torso_actual - 0) - c.KD_GAIN * torso_speed
        #control[self.stance_idx+c.HIP_OFFSET_TWIST] = -torso_torque# - control[self.swing_idx+c.HIP_OFFSET_TWIST]

        torques = self.env.from_features(control)
        return torques

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    #env.seed(133712)
    #env.seed(42)
    env.reset(random=0.0)
    env.sdf_loader.put_grounds([[0,0,0]], runway_length=20)
    for i in range(8):
        # TODO: for very small target steps (e.g. 10 cm), the velocity is so small that
        # the robot can get stuck in the UP state, balancing on one leg.
        t = 0.3 + 0.8*i# + np.random.uniform(low=-0.2, high=0.2)
        env.simulate([t,0,0], render=1, put_dots=True)
    embed()

