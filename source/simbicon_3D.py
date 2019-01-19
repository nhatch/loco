import numpy as np
from IPython import embed

from simbicon import Simbicon, UP, DOWN
from inverse_kinematics import InverseKinematics
from sdf_loader import RED, GREEN, BLUE
import utils

from consts_common3D import *
from simbicon_params import *

ZERO_GAIN = False

class Simbicon3D(Simbicon):

    def default_controllable_params(self):
        return [IK_GAIN, UP_IDX+SWING_ANKLE_RELATIVE, HEADING, POSITION_BALANCE_GAIN_LAT]

    def base_gait(self):
        gait = [0.14, 0.5, 0.2, -0.1, 0.2,
                0.4, -1.1,   0.0, -0.05,
                -0.0, -0.00, 0.20, -0.1,
                0.5, 0.2, 0.0, 0.0, 0.0, 0.0]
        return np.array(gait)

    def set_gait_raw(self, target, target_heading=None, raw_gait=None):
        if raw_gait is not None and self.swing_idx == LEFT_IDX:
            raw_gait = raw_gait * MIRROR_PARAMS
        return super().set_gait_raw(target, target_heading, raw_gait)

    def heading(self):
        return self.env.get_x()[0][ROOT_YAW]

    def balance_params(self, q, dq):
        theta = self.target_heading
        rot = utils.rotmatrix(theta)
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
        if ZERO_GAIN and d[Z]*v[Z] < 0:
            # If COM is moving (laterally) towards the stance heel, use no velocity gain.
            # This attempts to correct for a "sashay" issue that was causing self collisions.
            cv = 0
        balance_feedback = -(cd*d[Z] + cv*v[Z])

        tq[self.swing_idx+HIP_ROLL] = balance_feedback - q[ROOT_ROLL]
        tq[self.stance_idx+ANKLE_ROLL] = params[STANCE_ANKLE_ROLL]
        tq[self.swing_idx+ANKLE_ROLL] = params[SWING_ANKLE_ROLL]

        tq[TORSO_ROLL] = -q[ROOT_ROLL]

        self.update_doppelganger(tq)
        return tq

    def update_doppelganger(self, tq):
        dop = self.env.doppelganger
        if dop is None:
            return
        c = self.env.consts()
        tq = tq.copy()
        dop.q = self.env.from_features(tq)
        ik = InverseKinematics(self.env.doppelganger, self.env)
        offset = c.THIGH_BODYNODE_OFFSET
        dop_bodynode = ik.get_bodynode(self.stance_idx, offset)
        robot_bodynode = self.ik.get_bodynode(self.stance_idx, offset)
        tq[:6] = ik.get_dofs(robot_bodynode.transform(), dop_bodynode)
        self.env.doppelganger.q = self.env.from_features(tq)
        self.env.doppelganger.dq = np.zeros(c.Q_DIM)

def next_target(start, heading, length, env):
    c = env.controller
    rot = utils.rotmatrix(-heading)
    lateral_length = 0.3 if c.swing_idx == RIGHT_IDX else -0.3
    offset = [length, 0.0, lateral_length]
    target = start + np.dot(rot, offset)
    return target

def test(env, length, r=1, n=8, a=0.0, delta_a=0.0, relative=False, provide_target_heading=False):
    seed = np.random.randint(100000)
    obs = env.reset(seed=seed)
    #obs.mirror()
    obs.rotate(a)
    env.reset(obs, video_save_dir=None, render=r)
    env.sdf_loader.put_grounds([[-3.0,-0.9,0]])
    t = env.controller.stance_heel
    for i in range(n):
        l = length*0.5 if i == 0 else length
        t = next_target(t, a, l, env)
        target_heading = None
        if provide_target_heading:
            target_heading = a+delta_a
        _, terminated = env.simulate(t, target_heading=target_heading, put_dots=True)
        if relative:
            t = env.controller.stance_heel # Pretend that was the previous target
        a += delta_a
        if terminated:
            break

if __name__ == "__main__":
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv(Simbicon3D)
    env.sdf_loader.ground_width = 8.0
    # TODO: Get the controller to work well even when we don't provide the target heading.
    test(env, 0.5, delta_a=0.05, n=8, provide_target_heading=True)
    embed()
