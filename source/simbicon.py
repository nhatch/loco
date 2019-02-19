
from pd_control import PDController
import numpy as np
from IPython import embed
from inverse_kinematics import InverseKinematics
from utils import heading_from_vector

# The maximum time it should take to get (e.g.) the right foot off the ground
# after the left-foot heel strike.
LIFTOFF_DURATION = 0.3

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
            self.target = self.stance_heel
            swing_heel = self.ik.forward_kine(self.swing_idx)
            self.prev_target = swing_heel
        else:
            self.swing_idx = c.LEFT_IDX if state.swing_left else c.RIGHT_IDX
            self.stance_idx = c.RIGHT_IDX if state.swing_left else c.LEFT_IDX
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

    def base_gait(self):
        # Taken from Table 1 of https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf
        # Then modified for the new parameters format.
        gait = [0.14, 0, 0.2, 0.0, 0.2,
                0.4, -1.1,   0.0, -0.05,
                0,    0, 0.05, -0.1,
                0,0,0,0,0,0] # None of these last 6 are used in 2D
        return np.array(gait)

    def heading(self):
        return 0.0

    def set_gait_raw(self, target, target_heading=None, raw_gait=None):
        params = self.base_gait()
        if raw_gait is not None:
            params += raw_gait * sp.PARAM_SCALE
        self.set_gait(Params(params))

        self.target, self.prev_target = target, self.target

        swing_heel = self.ik.forward_kine(self.swing_idx)
        self.starting_swing_heel = swing_heel
        d = self.target - self.starting_swing_heel

        if target_heading is None:
            branch = self.heading()
            _, self.target_heading = heading_from_vector(d, branch)
            if self.env.is_3D:
                # Lateral balance feedback usually tends to put the swing foot too far "outside"
                # the target. Hack: adjust the target heading to roughly compensate for this.
                c = self.env.consts()
                dist = np.linalg.norm(d)
                adj = -dist*0.2
                if self.swing_idx == c.LEFT_IDX:
                    self.target_heading -= adj
                else:
                    self.target_heading += adj
        else:
            self.target_heading = target_heading
        if self.env.is_3D:
            self.target_heading += self.params[sp.HEADING]
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
        q, dq = self.env.get_x()
        _, v = self.balance_params(q, dq)
        if v[c.X] < -0.2:
            print("GOING BACKWARDS")
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
        # Technically in 3D this is a plane, not a line.
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
        if duration >= 0.2:
            # Once toe-off is complete, return to neutral ankle angle
            self.params[sp.SWING_ANKLE_RELATIVE+sp.UP_IDX] = self.base_gait()[sp.SWING_ANKLE_RELATIVE+sp.UP_IDX]
            if self.env.is_3D:
                self.params[sp.SWING_ANKLE_ROLL] = self.base_gait()[sp.SWING_ANKLE_ROLL]
        early_strike = (duration >= LIFTOFF_DURATION) and (len(contacts) > 0)
        #if early_strike:
        #    print("Early strike!")
        q, dq = self.env.get_x()
        target_diff = self.params[sp.IK_GAIN] * self.speed(dq)
        heel_close = self.distance_to_go(swing_heel) < target_diff
        com_close = self.distance_to_go(q[:3]) < target_diff
        if (heel_close and com_close) or early_strike:
            # Start the DOWN phase
            self.direction = DOWN

    def change_stance(self, contacts, swing_heel):
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
        tq[self.stance_idx+c.ANKLE] = params[sp.STANCE_ANKLE_RELATIVE]

        cd = params[sp.POSITION_BALANCE_GAIN]
        cv = params[sp.VELOCITY_BALANCE_GAIN]
        d, v = self.balance_params(q, dq)
        balance_feedback = cd*d[c.X] + cv*v[c.X]

        target_swing_angle = params[sp.SWING_HIP_WORLD+DIR_IDX] + balance_feedback
        target_swing_knee = params[sp.SWING_KNEE_RELATIVE+DIR_IDX]
        tq[self.swing_idx+c.KNEE] = target_swing_knee

        torso_actual = q[c.ROOT_PITCH]
        tq[self.swing_idx+c.HIP_PITCH] = target_swing_angle - torso_actual

        shin_actual = torso_actual + q[self.swing_idx+c.KNEE] + q[self.swing_idx+c.HIP_PITCH]
        # The following line sets the swing ankle to be flat relative to the ground.
        tq[self.swing_idx+c.ANKLE] = -shin_actual
        tq[self.swing_idx+c.ANKLE] += params[sp.SWING_ANKLE_RELATIVE+DIR_IDX]

        # This code is only useful in 3D.
        # The stance hip pitch torque will be overwritten in `compute` below.
        target_orientation = self.ik.root_transform_from_angles(self.target_heading, params[sp.TORSO_WORLD])
        hip_dofs = self.ik.get_hip(self.stance_idx, target_orientation)
        if self.env.is_3D:
            tq[self.stance_idx:self.stance_idx+3] = hip_dofs
            # This is a hack; the hip was dipping a little bit too much on the swing side.
            # The proper fix: rather than just using kinematics to set the target angles,
            # also compensate for the torques from other forces on the pelvis.
            tq[self.stance_idx+c.HIP_ROLL] += 0.1 if self.stance_idx == c.LEFT_IDX else -0.1
        else:
            tq[self.stance_idx] = hip_dofs

        return tq

    def compute(self):
        c = self.env.consts()
        q, dq = self.env.get_x()
        params = self.params
        target_q = self.compute_target_q(q, dq)
        self.update_doppelganger(target_q)

        # We briefly increase Kd (mechanical impedance?) for the stance knee
        # in order to prevent the robot from stepping so hard that it bounces.
        fix_Kd = self.direction == UP and self.time() - self.step_started < 0.1
        fix_Kd_idx = c.fix_Kd_idx(self.stance_idx)
        if fix_Kd:
            # TODO this Kd index is wrong!!
            self.Kd[fix_Kd_idx] *= 8
        raw_control = self.compute_PD(target_q)
        if fix_Kd:
            self.Kd[fix_Kd_idx] /= 8

        # Make modifications to control torso pitch
        control = c.standardized_dofs(raw_control)
        torso_actual = q[c.ROOT_PITCH]
        torso_speed = dq[c.ROOT_PITCH]
        virtual_torque_idx = c.virtual_torque_idx(self.stance_idx)
        kp = self.Kp[virtual_torque_idx]
        kd = self.Kd[virtual_torque_idx]
        torso_torque = - kp * (torso_actual - params[sp.TORSO_WORLD]) - kd * torso_speed
        control[self.stance_idx+c.HIP_PITCH] = -torso_torque - control[self.swing_idx+c.HIP_PITCH]
        transformed_control = c.raw_dofs(control)
        raw_control[virtual_torque_idx] = transformed_control[virtual_torque_idx]

        return raw_control

    def update_doppelganger(self, tq):
        dop = self.env.doppelganger
        if dop is None:
            return
        c = self.env.consts()
        tq = tq.copy()
        dop.q = c.raw_dofs(tq)
        ik = InverseKinematics(self.env.doppelganger, self.env)
        offset = c.THIGH_BODYNODE_OFFSET
        dop_bodynode = ik.get_bodynode(self.stance_idx, offset)
        robot_bodynode = self.ik.get_bodynode(self.stance_idx, offset)
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

def test(env, length, n=8, seed=None, runway_length=15, runway_x=0, render=1, video_save_dir=None):
    env.clear_skeletons() # Necessary in order to change the runway length
    env.sdf_loader.ground_length = runway_length
    start_state = env.reset(seed=seed, random=0.005,
            render=render, video_save_dir=video_save_dir)
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
    #test(env, 0.5, n=4, render=2, video_save_dir='monitoring')
    test(env, 0.5)
    #reproduce_bug(env)
    embed()
