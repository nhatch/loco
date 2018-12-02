
from pd_control import PDController
import numpy as np
from IPython import embed
from inverse_kinematics import InverseKinematics

# The maximum time it should take to get (e.g.) the right foot off the ground
# after the left-foot heel strike.
LIFTOFF_DURATION = 0.3

from simbicon_params import *

UP = 'UP'
DOWN = 'DOWN'

class Params:
    def __init__(self, params):
        self.raw_params = params

    def __getitem__(self, key):
        return self.raw_params[key]

    def __setitem__(self, key, value):
        self.raw_params[key] = value

class Simbicon(PDController):

    def __init__(self, skel, env):
        self.ik = InverseKinematics(env)
        super().__init__(skel, env)

    def reset(self, state=None):
        c = self.env.consts()
        self.step_started = self.time()
        self.swing_idx = c.RIGHT_IDX
        self.stance_idx = c.LEFT_IDX
        self.direction = UP

        if state is None:
            self.stance_heel = self.ik.forward_kine(self.stance_idx)
            self.target = np.zeros(3)
            self.prev_target = np.zeros(3)
        else:
            self.stance_heel = state.stance_heel_location()
            # The method `set_gait_raw` must also be called before simulation continues,
            # in order to set a new target and gait.
            # At that point, we will rotate self.target to self.prev_target.
            self.target = state.stance_platform()
            # We track prev_target so that, in principle, we know all of the possible
            # places where the agent is in contact with the environment. However, the
            # algorithm does not use this (yet) except when resetting the environment
            # to a previously visited state.
            self.prev_target = state.swing_platform()

    def state(self):
        # TODO make State attributes settable, so we can construct this like
        # state.stance_platform = self.target
        # Until then, make sure that this order corresponds to the indices defined
        # in the State object.
        return np.concatenate((self.stance_heel, self.target, self.prev_target))

    def standardize_stance(self, state):
        # We need to flip the left and right leg.
        c = self.env.consts()
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
        gait = [0.14, 0, 0.2, 0.0, 0.2,
                0.4, -1.1,   0, -0.05,
                0,    0, 0.2, -0.1,
                0,0,0]
        return gait

    def controllable_indices(self):
        return np.array([1, 1, 1, 1, 1,
                         0, 0, 0, 0,
                         0, 0, 0, 1,
                         1,1,0])

    def set_gait_raw(self, target, raw_gait=None):
        params = self.base_gait()
        if raw_gait is not None:
            params += raw_gait
        self.set_gait(Params(params))

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

        self.adjust_targets()

    def adjust_targets(self):
        # All of these adjustments are just rough linear estimates from
        # fiddling around manually.
        params = self.params # This should work like np.array
        params[TX] += self.target[0]
        params[TY] += self.target[1]
        params[TZ] += self.target[2]
        step_dist_diff = params[TX] - self.stance_heel[0] - 0.4
        params[STANCE_KNEE_RELATIVE+UP_IDX]  += - step_dist_diff * 0.2
        params[SWING_HIP_WORLD+UP_IDX]       += + step_dist_diff * 0.4
        params[SWING_KNEE_RELATIVE+UP_IDX]   += - step_dist_diff * 0.8
        params[STANCE_ANKLE_RELATIVE] += + step_dist_diff * 0.4
        params[TORSO_WORLD]           += - step_dist_diff * 0.5
        #q, dq = self.env.get_x()
        #tq = self.compute_target_q(q, dq)
        #print("Ending swing roll (world):", q[self.stance_idx + 2]+q[5])
        #print("==========================")
        #print("Target swing roll (world):", tq[self.swing_idx + 2]+q[5])
        #print("Actual swing roll (world):", q[self.swing_idx + 2]+q[5])
        #print("Lateral offset from stance heel:", q[2] - self.stance_heel[2])
        #print("Lateral velocity:", dq[2])

    def set_gait(self, params):
        self.params = params

    def time(self):
        return self.env.world.time()

    def crashed(self, swing_heel):
        c = self.env.consts()
        for contact in self.env.world.collision_result.contacts:
            if contact.skel_id1 == 1:
                bodynode = contact.bodynode1
            elif contact.skel_id2 == 1:
                bodynode = contact.bodynode2
            else:
                continue
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

    # Returns True if the swing foot has made contact.
    def swing_contact(self, contacts, swing_heel):
        if self.direction == UP:
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
        target_diff = self.params[IK_GAIN] * dq[0]
        heel_close = self.params[TX] < swing_heel[0] + target_diff
        # TODO somehow this is misfiring, I think. TODO debug `python simbicon.py`
        com_close = self.params[TX] < q[0] + target_diff
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
        res = "({:.2f}, {:.2f}, {:.2f}) ({:+.2f}, {:+.2f}, {:.2f})".format(hx, hy, hz, dx, dy, dz)
        return "{:.3f}: Ended step at {}".format(self.time(), res)

    def calc_down_ik(self):
        c = self.env.consts()
        q, dq = self.env.get_x()
        # Upon starting the DOWN part of the step, choose target swing leg angles
        # based on the location on the ground at target.
        tx = self.params[TX] - self.params[IK_GAIN] * dq[0]
        ty = self.params[TY] - 0.1 # TODO should we adjust this based on vertical velocity?
        relative_hip, knee = self.ik.inv_kine([tx, ty])
        # TODO this completely overrides the base_gait and set_gait settings. Make this clearer
        self.params[SWING_HIP_WORLD+DN_IDX] = relative_hip + q[c.ROOT_PITCH]
        self.params[SWING_KNEE_RELATIVE+DN_IDX] = knee

    def compute_target_q(self, q, dq):
        c = self.env.consts()
        params = self.params
        DIR_IDX = UP_IDX if self.direction == UP else DN_IDX
        tq = np.zeros(c.Q_DIM)
        tq[self.stance_idx+c.KNEE] = params[STANCE_KNEE_RELATIVE+DIR_IDX]
        tq[self.stance_idx+c.ANKLE] = params[STANCE_ANKLE_RELATIVE]

        cd = params[POSITION_BALANCE_GAIN]
        cv = params[VELOCITY_BALANCE_GAIN]
        v = dq[c.X]
        d = q[c.X] - self.stance_heel[c.X]
        balance_feedback = cd*d + cv*v

        target_swing_angle = params[SWING_HIP_WORLD+DIR_IDX] + balance_feedback
        target_swing_knee = params[SWING_KNEE_RELATIVE+DIR_IDX]
        tq[self.swing_idx+c.KNEE] = target_swing_knee

        # The following line sets the swing ankle to be flat relative to the ground.
        tq[self.swing_idx+c.ANKLE] = -(target_swing_angle + target_swing_knee)
        tq[self.swing_idx+c.ANKLE] += params[SWING_ANKLE_RELATIVE+DIR_IDX]

        torso_actual = q[c.ROOT_PITCH]
        tq[self.swing_idx+c.HIP_PITCH] = target_swing_angle - torso_actual
        return tq

    def compute(self):
        c = self.env.consts()
        q, dq = self.env.get_x()
        params = self.params
        target_q = self.compute_target_q(q, dq)
        # We briefly increase Kd (mechanical impedance?) for the stance knee
        # in order to prevent the robot from stepping so hard that it bounces.
        fix_Kd = self.direction == UP and self.time() - self.step_started < 0.1
        if fix_Kd:
            self.Kd[self.stance_idx+c.KNEE] *= 8
        control = self.compute_transformed(target_q)
        if fix_Kd:
            self.Kd[self.stance_idx+c.KNEE] /= 8

        # Make modifications to control torso pitch
        torso_actual = q[c.ROOT_PITCH]
        torso_speed = dq[c.ROOT_PITCH]
        kp = self.Kp[self.stance_idx+c.HIP_PITCH]
        kd = self.Kd[self.stance_idx+c.HIP_PITCH]
        torso_torque = - kp * (torso_actual - params[TORSO_WORLD]) - kd * torso_speed
        control[self.stance_idx] = -torso_torque - control[self.swing_idx]

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
        # TODO: for long steps, (e.g. 80 cm) the robot hits the target with its toe rather
        # than its heel. This makes difficult training environments for random optimization.
        t = 0.3 + 0.8*i# + np.random.uniform(low=-0.2, high=0.2)
        env.simulate([t,0,0], render=1, put_dots=True)
    embed()

