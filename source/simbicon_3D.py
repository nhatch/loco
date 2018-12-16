import numpy as np
from IPython import embed

from simbicon import Simbicon, UP, DOWN
from inverse_kinematics import InverseKinematics
from state import State
from sdf_loader import RED, GREEN, BLUE

from consts_common3D import *
from simbicon_params import *

class Simbicon3D(Simbicon):
    def standardize_stance(self, state):
        c = self.env.consts()
        state = super().standardize_stance(state)
        # Rotations are absolute, not relative, so we need to multiply some angles
        # by -1 to obtain a mirrored pose.
        D = Q_DIM
        m = np.ones(D)
        absolute_rotation_indices = [Z, ROOT_YAW, ROOT_ROLL,
                RIGHT_IDX + HIP_YAW, RIGHT_IDX + HIP_ROLL, RIGHT_IDX + ANKLE_ROLL,
                LEFT_IDX + HIP_YAW, LEFT_IDX + HIP_ROLL, LEFT_IDX + ANKLE_ROLL,
                TORSO_ROLL, TORSO_YAW]
        m[absolute_rotation_indices] = -1
        state[0:D] *= m
        state[D:2*D] *= m
        state[[-7,-4,-1]] *= -1 # Z coordinates of heel location and platforms
        return state

    def base_gait(self):
        gait = [0.14, 0.5, 0.2, -0.1, 0.2,
                0.4, -1.1,   0.0, -0.05,
                -0.0, -0.00, 0.20, -0.1,
                0,0,0,
                0.7, 0.2, 0.0, 0.0, 0.0]
        return gait

    def controllable_indices(self):
        return np.array([1, 0, 0, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 0,
                         0,0,0,
                         1, 1, 1, 0, 1])

    def heading(self, q):
        return q[ROOT_YAW]

    def rotmatrix(self, theta):
        # Note we're rotating in the X-Z plane instead of X-Y, so some signs are weird.
        return np.array([[np.cos(theta), 0, -np.sin(theta)],
                         [            0, 1,              0],
                         [np.sin(theta), 0,  np.cos(theta)]])

    def balance_params(self, q, dq):
        theta = self.heading(q)
        rot = self.rotmatrix(theta)
        com_displacement = np.dot(rot, q[:3] - self.stance_heel)
        heading_velocity = np.dot(rot, dq[:3])
        return com_displacement, heading_velocity

    def compute_target_q(self, q, dq):
        tq = super().compute_target_q(q, dq)

        c = self.env.consts()
        params = self.params

        cd = params[POSITION_BALANCE_GAIN_LAT]
        cv = params[VELOCITY_BALANCE_GAIN_LAT]

        proj = q[:3]
        proj[Y] = self.stance_heel[Y]
        self.env.sdf_loader.put_dot(proj, 'root_projection', color=BLUE)
        self.env.sdf_loader.put_dot(self.stance_heel, 'stance_heel', color=RED)

        d, v = self.balance_params(q, dq)
        if d[Z]*v[Z] < 0:
            # If COM is moving (laterally) towards the stance heel, use no velocity gain.
            # This attempts to correct for a "sashay" issue that was causing self collisions.
            cv = 0
        balance_feedback = -(cd*d[Z] + cv*v[Z])

        # TODO this code is WIP; doesn't do anything yet
        k_yaw = 0.2
        rot = self.rotmatrix(self.heading(q))
        target_lateral_offset = np.dot(rot, self.target - self.stance_heel)[Z]
        if self.stance_idx == RIGHT_IDX:
            target_lateral_offset *= -1
        target_yaw = -k_yaw*(target_lateral_offset - 0.3) # relative to current heading
        #tq[self.stance_idx+HIP_YAW] = q[self.stance_idx+HIP_YAW]+target_yaw
        #tq[self.swing_idx+HIP_YAW] = q[self.swing_idx+HIP_YAW]+target_yaw

        tq[self.swing_idx+HIP_ROLL] = balance_feedback - q[ROOT_ROLL]
        tq[self.stance_idx+HIP_ROLL] = params[STANCE_HIP_ROLL_EXTRA] + q[ROOT_ROLL]
        tq[self.stance_idx+ANKLE_ROLL] = params[STANCE_ANKLE_ROLL]

        tq[TORSO_ROLL] = -q[ROOT_ROLL]

        self.update_doppelganger(tq)
        return tq

    def post_process(self, control, q, dq):
        control = super().post_process(control, q, dq)
        p = self.target_heading - q[ROOT_YAW]
        d = 0.0 - dq[ROOT_YAW]
        kp = self.Kp[self.stance_idx+HIP_YAW]
        kd = self.Kd[self.stance_idx+HIP_YAW]
        control[self.stance_idx+HIP_YAW] = -(p*kp + d*kd)
        return control

    def update_doppelganger(self, tq):
        if self.env.doppelganger is None:
            return
        c = self.env.consts()
        tq = tq.copy()
        tq[ROOT_PITCH] = self.params[TORSO_WORLD]
        tq[ROOT_YAW] = self.target_heading
        q = self.env.get_x()[0]
        tq[ROOT_ROLL] = q[ROOT_ROLL] # We don't try to control this (yet?)
        self.env.doppelganger.q = self.env.from_features(tq)
        self.env.doppelganger.dq = np.zeros(c.Q_DIM)
        ik = InverseKinematics(self.env.doppelganger, self.env)
        stance_heel = ik.forward_kine(self.stance_idx)
        heel_fixing_trans = self.stance_heel - stance_heel
        offset_trans = np.array([0.0, 0.0, 1.5])
        tq[:3] += heel_fixing_trans + offset_trans
        self.env.doppelganger.q = self.env.from_features(tq)

def test_standardize_stance(env):
    from time import sleep
    env.reset(random=0.5)
    env.set_rot_manual(np.pi/2)
    c = env.controller
    c.change_stance([], [0,0,0])
    obs = env.current_observation()
    env.render()
    sleep(0.5)
    env.reset(obs, random=0.0)
    env.render()

def next_target(start, heading, length, env):
    c = env.controller
    rot = c.rotmatrix(-heading)
    lateral_length = 0.3 if c.swing_idx == RIGHT_IDX else -0.3
    offset = [length, 0.0, lateral_length]
    target = start + np.dot(rot, offset)
    return target

def rotate_state(state, angle, env):
    rotated = State(state.raw_state.copy())
    rotated.raw_state[ROOT_YAW] += angle
    rot = env.controller.rotmatrix(-angle)
    rotated.pose()[:3] = np.dot(rot, state.pose()[:3])
    rotated.dq()[:3] = np.dot(rot, state.dq()[:3])
    rotated.stance_heel_location()[:3] = np.dot(rot, state.stance_heel_location()[:3])
    rotated.stance_platform()[:3] = np.dot(rot, state.stance_platform()[:3])
    rotated.swing_platform()[:3] = np.dot(rot, state.swing_platform()[:3])
    return rotated

def test(env, length, r=1, n=8, a=0.0):
    seed = np.random.randint(100000)
    obs = env.reset(seed=seed, random=0)
    env.reset(rotate_state(obs, a, env), random=0.0)
    env.sdf_loader.put_grounds([[-30.0,-0.9,0]], runway_length=40.0)
    t = env.controller.stance_heel
    for i in range(n):
        l = length*0.5 if i == 0 else length
        a = 0.07*(i+1)
        t = next_target(env.controller.stance_heel, a, l, env)
        _, terminated = env.simulate(t, render=r, put_dots=True)
        if terminated:
            break

if __name__ == "__main__":
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv(Simbicon3D)
    env.sdf_loader.ground_width = 60.0
    test(env, 0.6, n=200, r=1)
    #test_standardize_stance(env)
    embed()
