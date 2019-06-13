
from pd_control import PDController
import numpy as np
from IPython import embed
from inverse_kinematics import InverseKinematics
from utils import heading_from_vector

import simbicon_params as sp

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

    def __init__(self, env):
        super().__init__(env)
        if hasattr(env, 'world'):
            self.set_RRTs()

    def reset(self, state=None):
        c = self.env.consts()
        self.ik = InverseKinematics(self.env.robot_skeleton, self.env)
        self.step_started = self.time()
        self.direction = UP

        if state is None:
            self.swing_idx = c.RIGHT_IDX
            self.stance_idx = c.LEFT_IDX
            self.stance_heel = self.ik.forward_kine(self.stance_idx)
            self.stance_platform = self.stance_heel
            swing_heel = self.ik.forward_kine(self.swing_idx)
            self.swing_platform = swing_heel
            self.target = swing_heel # For Darwin's sake. This is never set, otherwise.
        else:
            self.swing_idx = c.LEFT_IDX if state.swing_left else c.RIGHT_IDX
            self.stance_idx = c.RIGHT_IDX if state.swing_left else c.LEFT_IDX
            self.stance_heel = state.stance_heel_location()
            self.stance_platform = state.stance_platform()
            # We track swing_platform so that, in principle, we know all of the possible
            # places where the agent is in contact with the environment. However, the
            # algorithm does not use this (yet) except when resetting the environment
            # to a previously visited state.
            self.swing_platform = state.swing_platform()
        # The method `set_gait_raw` must also be called before simulation continues,
        # in order to set a target and gait.

    def state(self):
        # TODO make State attributes settable, so we can construct this like
        # state.stance_platform = self.stance_platform
        # Until then, make sure that this order corresponds to the indices defined
        # in the State object.
        return np.concatenate((self.stance_heel, self.stance_platform, self.swing_platform))

    def heading(self):
        return 0.0

    def set_gait_raw(self, target, target_heading=None, raw_gait=None):
        c = self.env.consts()
        params = c.BASE_GAIT.copy()
        if raw_gait is not None:
            if len(raw_gait) < sp.N_PARAMS:
                # For backwards compatibility (if we add new Simbicon params later)
                raw_gait = np.concatenate((raw_gait, [0.]*(sp.N_PARAMS - len(raw_gait))))
            params += raw_gait * sp.PARAM_SCALE
        self.params = params

        if not c.OBSERVE_TARGET:
            self.unit_normal = np.array([0., 1., 0.])
            return # Skip the rest of setup

        swing_heel = self.ik.forward_kine(self.swing_idx)
        self.target = target
        d = self.target - swing_heel

        if target_heading is None:
            branch = self.heading()
            self.target_heading = branch - self.params[sp.STANCE_YAW]
        else:
            self.target_heading = target_heading
        self.target_direction = np.array(
                [np.cos(self.target_heading), 0, -np.sin(self.target_heading)])

        # Gives the vector in the ground plane perpendicular to the direction `d`
        # such that the cross product between those two vectors should point up-ish.
        transverse_d = np.array([d[2], 0.0, -d[0]])
        # If d=[x,y,z] the result should be [-xy, z^2+x^2, -yz]
        d = np.cross(d, transverse_d)
        # Calculate a normal vector for the line between the starting location
        # of the swing heel and the target.
        # This is used when detecting crashes and evaluating progress of
        # the learning algorithm.
        self.unit_normal = d / np.linalg.norm(d)

        self.adjust_targets()

    def distance_to_go(self, current_location):
        # TODO use adjusted target not self.target
        d = np.dot(self.target_direction, self.target - current_location)
        return d

    def speed(self, dq):
        v = np.dot(self.target_direction, dq[:3])
        return v

    def adjust_targets(self):
        # All of these adjustments are just rough linear estimates from
        # fiddling around manually.
        params = self.params # This should work like np.array
        step_dist_diff = self.distance_to_go(self.stance_heel) - 0.4
        params[sp.STANCE_KNEE_RELATIVE+sp.UP_IDX]  += - step_dist_diff * 0.2
        params[sp.SWING_HIP_WORLD+sp.UP_IDX]       += + step_dist_diff * 0.4
        params[sp.SWING_KNEE_RELATIVE+sp.UP_IDX]   += - step_dist_diff * 0.8
        params[sp.STANCE_ANKLE_RELATIVE] += + step_dist_diff * 0.4
        params[sp.TORSO_WORLD]           += - step_dist_diff * 0.5


    def time(self):
        return self.env.time()

    def crashed(self, swing_heel):
        # For some reason, setting the tolerance smaller than .05 or so causes the controller
        # to learn very weird behaviors. TODO: why does this have such a large effect??
        # However, setting the tolerance too large (larger than .04 or so) makes certain crashes
        # go undetected.
        tol = -0.06
        lower = min(self.swing_platform[1], self.target[1])
        upper = max(self.swing_platform[1], self.target[1])
        below_lower = swing_heel[1] - lower < tol
        below_upper = swing_heel[1] - upper < tol
        # Calculate the distance of the swing heel from the line between the
        # start and end targets.
        # Technically in 3D this is a plane, not a line.
        below_line = np.dot(swing_heel - self.swing_platform, self.unit_normal) < tol
        if below_lower or (below_line and below_upper):
            print("FELL OFF")
            return True

    # Returns True if the swing foot has made contact.
    def swing_contact(self, contacts, swing_heel):
        if self.direction == UP:
            return self.maybe_start_down_phase(contacts, swing_heel)
        elif len(contacts) > 0: # and self.direction == DOWN
            return True
        else:
            return False

    def maybe_start_down_phase(self, contacts, swing_heel):
        c = self.env.consts()
        duration = self.time() - self.step_started
        if c.OBSERVE_TARGET:
            if duration >= 0.2:
                # Once toe-off is complete, return to neutral ankle angle
                self.params[sp.SWING_ANKLE_RELATIVE+sp.UP_IDX] = c.BASE_GAIT[sp.SWING_ANKLE_RELATIVE+sp.UP_IDX]
                if self.env.is_3D:
                    self.params[sp.SWING_ANKLE_ROLL] = c.BASE_GAIT[sp.SWING_ANKLE_ROLL]
            early_strike = (duration >= c.LIFTOFF_DURATION) and (len(contacts) > 0)
            q, dq = self.env.get_x()
            target_diff = self.params[sp.IK_GAIN] * self.speed(dq)
            heel_close = self.distance_to_go(swing_heel) < target_diff
            com_close = self.distance_to_go(q[:3]) < target_diff
            start_down = early_strike or (heel_close and com_close)
            if start_down:
                self.direction = DOWN
        else:
            # Avoid DOWN phase entirely; don't perceive footstrikes
            early_strike = duration >= self.params[sp.UP_DURATION]
        return early_strike

    def change_stance(self, swing_heel):
        self.swing_platform, self.stance_platform = self.stance_platform, self.target
        self.step_started = self.time()
        self.stance_heel = swing_heel
        self.direction = UP
        self.swing_idx, self.stance_idx = self.stance_idx, self.swing_idx
        hx, hy, hz = self.stance_heel
        dx, dy, dz = self.stance_heel - self.target
        res = "({:.2f}, {:.2f}, {:.2f}) ({:+.2f}, {:+.2f}, {:.2f})".format(hx, hy, hz, dx, dy, dz)
        return "{:.3f}: Ended step at {}".format(self.time(), res)

    def balance_params(self, q, dq):
        return q[:1] - self.stance_heel[:1], dq[:1] # Take just the X coordinate

    def compute_target_q(self, q, dq):
        c = self.env.consts()
        params = self.params
        DIR_IDX = sp.UP_IDX if self.direction == UP else sp.DN_IDX
        tq = np.zeros(c.Q_DIM)
        tq[self.stance_idx+c.KNEE] = params[sp.STANCE_KNEE_RELATIVE+DIR_IDX]

        cd = params[sp.POSITION_BALANCE_GAIN]
        cv = params[sp.VELOCITY_BALANCE_GAIN]
        d, v = self.balance_params(q, dq)
        balance_feedback = cd*d[c.X] + cv*v[c.X]

        target_swing_angle = params[sp.SWING_HIP_WORLD+DIR_IDX] + balance_feedback
        target_swing_knee = params[sp.SWING_KNEE_RELATIVE+DIR_IDX]
        tq[self.swing_idx+c.KNEE] = target_swing_knee

        torso_actual = q[c.ROOT_PITCH]
        tq[self.swing_idx+c.HIP_PITCH] = target_swing_angle - torso_actual

        if c.OBSERVE_TARGET:
            # The following sets the ankles to be flat relative to the ground.
            ANKLE_DOF = 2 if self.env.is_3D else 1
            tq[self.swing_idx+c.ANKLE:self.swing_idx+c.ANKLE+ANKLE_DOF] = self.ik.get_ankle(self.swing_idx)
            tq[self.stance_idx+c.ANKLE:self.stance_idx+c.ANKLE+ANKLE_DOF] = self.ik.get_ankle(self.stance_idx)
            # This overwrites parallel-to-ground correction for stance ankle pitch.
            # I guess I'd rather not do this, but fixing this "bug" breaks the controller
            # for the simple 2D and 3D models.
            tq[self.stance_idx+c.ANKLE] = 0
        tq[self.swing_idx+c.ANKLE] += params[sp.SWING_ANKLE_RELATIVE+DIR_IDX]
        tq[self.stance_idx+c.ANKLE] += params[sp.STANCE_ANKLE_RELATIVE]

        # This code is only useful in 3D.
        # The stance hip pitch torque will be overwritten in `compute` below.
        if c.OBSERVE_TARGET:
            target_orientation = self.ik.root_transform_from_angles(self.target_heading, params[sp.TORSO_WORLD])
            hip_dofs = self.ik.get_hip(self.stance_idx, target_orientation)
            if self.env.is_3D:
                tq[self.stance_idx:self.stance_idx+3] = hip_dofs
                # It turns out that this "virtual target" for stance hip angles causes
                # extreme instability when the stance foot does not have enough friction
                # with the ground. To fix this, we ignore the virtual target for the yaw
                # angle and instead directly set that target using a parameter.
                # TODO: This breaks the ability of Simbicon3D to turn (i.e. follow
                # a target heading). Can we repair that by using this parametrization in
                # a creative way?
                # TODO: Since we're overwriting 2/3 stance hip angles, it probably makes
                # sense just to remove the inverse-kinematics-based stance hip angle
                # calculation completely.
                tq[self.stance_idx+c.HIP_YAW] = params[sp.STANCE_YAW]
                # This is a hack; the hip was dipping a little bit too much on the swing side.
                # The proper fix: rather than just using kinematics to set the target angles,
                # also compensate for the torques from other forces on the pelvis.
                tq[self.stance_idx+c.HIP_ROLL] += 0.1 if self.stance_idx == c.LEFT_IDX else -0.1
            else:
                tq[self.stance_idx] = hip_dofs
        if self.env.is_3D:
            # Don't use hip_dofs. (We probably won't have good enough sensors on hardware.)
            extra_roll = params[sp.STANCE_HIP_ROLL_EXTRA]
            tq[self.stance_idx+c.HIP_ROLL] += extra_roll
            #tq[self.swing_idx+c.HIP_ROLL] -= extra_roll

        # Make modifications to control torso pitch
        virtual_torque_idx = c.virtual_torque_idx(self.stance_idx)
        # Should be the same for both stance and swing
        kp = self.Kp[virtual_torque_idx]
        kd = self.Kd[virtual_torque_idx]
        # I say "probable" because I'm not sure the robot will calculate things quite this way
        probable_swing_torque = kp * (tq[self.swing_idx+c.HIP_PITCH] - q[self.swing_idx+c.HIP_PITCH]) - kd * dq[self.swing_idx+c.HIP_PITCH]
        torso_torque = - kp * (q[c.ROOT_PITCH] - params[sp.TORSO_WORLD]) - kd * dq[c.ROOT_PITCH]
        desired_stance_torque = -torso_torque - probable_swing_torque
        tq_stance = (desired_stance_torque + kd*dq[self.stance_idx+c.HIP_PITCH]) / kp + q[self.stance_idx+c.HIP_PITCH]
        tq[self.stance_idx+c.HIP_PITCH] = tq_stance

        return tq

    def compute(self):
        c = self.env.consts()
        q, dq = self.env.get_x()
        if self.env.world.frame % c.FRAMES_PER_CONTROL == 0:
            self.target_q = c.clip(c.raw_dofs(self.compute_target_q(q, dq)))
        self.update_doppelganger()

        # We briefly increase Kd (mechanical impedance?) for the stance knee
        # in order to prevent the robot from stepping so hard that it bounces.
        fix_Kd = (not self.env.is_3D) and self.direction == UP \
                and self.time() - self.step_started < 0.1
        if fix_Kd:
            fix_Kd_idx = c.fix_Kd_idx(self.stance_idx)
            self.Kd[fix_Kd_idx] *= 8
        raw_control = self.compute_PD()
        if fix_Kd:
            self.Kd[fix_Kd_idx] /= 8

        return raw_control

    def update_doppelganger(self):
        dop = self.env.doppelganger
        if dop is None:
            return
        c = self.env.consts()
        dop.q = self.target_q
        tq = c.standardized_dofs(self.target_q)
        ik = InverseKinematics(self.env.doppelganger, self.env)
        dop_bodynode = ik.root_bodynode()
        robot_bodynode = self.ik.root_bodynode()
        tq[:c.BRICK_DOF] = ik.get_dofs(robot_bodynode.transform(), dop_bodynode)
        dop.q = c.raw_dofs(tq)
        dop.dq = np.zeros(c.Q_DIM_RAW)

    def set_RRTs(self):
        c = self.env.consts()
        thigh_r = self.ik.get_bodynode(c.RIGHT_IDX, c.THIGH_BODYNODE_OFFSET)
        thigh_l = self.ik.get_bodynode(c.LEFT_IDX, c.THIGH_BODYNODE_OFFSET)
        pelvis = self.ik.root_bodynode()
        # The relative transform of the thigh when all DOFs of the joint are set to zero
        # (inverted to save computation--we only ever use the inverse)
        c.LEFT_RRT_INV = np.dot(np.linalg.inv(thigh_l.transform()), pelvis.transform())
        c.RIGHT_RRT_INV = np.dot(np.linalg.inv(thigh_r.transform()), pelvis.transform())

        foot_r = self.ik.get_bodynode(c.RIGHT_IDX, c.FOOT_BODYNODE_OFFSET)
        foot_l = self.ik.get_bodynode(c.LEFT_IDX, c.FOOT_BODYNODE_OFFSET)
        shin_r = self.ik.get_bodynode(c.RIGHT_IDX, c.SHIN_BODYNODE_OFFSET)
        shin_l = self.ik.get_bodynode(c.LEFT_IDX, c.SHIN_BODYNODE_OFFSET)
        c.LEFT_RRT_INV_ANKLE = np.dot(np.linalg.inv(foot_l.transform()), shin_l.transform())
        c.RIGHT_RRT_INV_ANKLE = np.dot(np.linalg.inv(foot_r.transform()), shin_r.transform())

def test(env, length, n=8, seed=None, runway_length=15, runway_x=0, r=1, video_save_dir=None):
    env.clear_skeletons() # Necessary in order to change the runway length
    env.sdf_loader.ground_length = runway_length
    start_state = env.reset(seed=seed, random=0.005,
            render=r, video_save_dir=video_save_dir)
    env.sdf_loader.put_grounds([[runway_x,0,0]])
    action = np.zeros(sp.N_PARAMS)
    for i in range(n):
        #if i > 2:
        #    action[sp.UP_IDX+sp.SWING_ANKLE_RELATIVE] = -0.3
        t = length*(0.5 + i)# + np.random.uniform(low=-0.2, high=0.2)
        _, terminated = env.simulate([t,0,0], action=action, put_dots=True)
        if terminated:
            break

def reproduce_bug(env):
    # Don't abort the simulation if the shins touch the ground.
    # This makes the bug more obvious visually (otherwise the only evidence will be
    # that the simulation stops early and writes "ERROR: Crashed" to the console).
    env.consts().ALLOWED_COLLISION_IDS.append(4)
    env.consts().ALLOWED_COLLISION_IDS.append(7)

    seed=73298 # Pretty much any seed will do
    test(env, 0.6, seed=seed, runway_length=100)
    embed()
    # But note that if runway_x is -50 then this still works.
    test(env, 0.6, seed=seed, runway_length=100, runway_x=-50)
    # And if the runway is shorter then it also still works.
    test(env, 0.6, seed=seed, runway_length=15)

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    test(env, 0.5)
    embed()
