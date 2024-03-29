import numpy as np
from IPython import embed

from simbicon import Simbicon
from inverse_kinematics import InverseKinematics
from sdf_loader import RED, GREEN, BLUE
import utils

from consts_common3D import *
import simbicon_params as sp

ZERO_GAIN = False

class Simbicon3D(Simbicon):
    def set_gait_raw(self, target, raw_gait=None):
        if raw_gait is not None and self.swing_idx == LEFT_IDX:
            raw_gait = raw_gait * sp.MIRROR_PARAMS
        return super().set_gait_raw(target, raw_gait)

    def heading(self):
        return self.env.get_x()[0][ROOT_YAW]

    def balance_params(self, q, dq):
        c = self.env.consts()
        if c.OBSERVE_TARGET:
            theta = self.target_heading
            rot = utils.rotmatrix(theta)
            com_displacement = np.dot(rot, q[:3] - self.stance_heel)
            heading_velocity = np.dot(rot, dq[:3])
            return com_displacement, heading_velocity
        else:
            # return np.zeros(3), np.zeros(3) # Oddly, this works fine.
            # On the Darwin hardware, it may be easier to use the direct IMU measurements
            # Divide by five to avoid triggering the "GOING BACKWARDS" crash detector
            d = np.array([-q[c.ROOT_PITCH], 0, q[c.ROOT_ROLL]]) / 5
            v = np.array([-dq[c.ROOT_PITCH], 0, dq[c.ROOT_ROLL]]) / 5
            return d,v

    def compute_target_q(self, q, dq):
        tq = super().compute_target_q(q, dq)

        c = self.env.consts()
        params = self.params

        cd = params[sp.POSITION_BALANCE_GAIN_LAT]
        cv = params[sp.VELOCITY_BALANCE_GAIN_LAT]

        if hasattr(self.env, 'sdf_loader'):
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

        tq[self.swing_idx+HIP_ROLL] += balance_feedback - q[ROOT_ROLL]
        tq[self.stance_idx+ANKLE_ROLL] += params[sp.STANCE_ANKLE_ROLL]
        tq[self.swing_idx+ANKLE_ROLL] += params[sp.SWING_ANKLE_ROLL]

        tq[TORSO_ROLL] = -q[ROOT_ROLL] + params[sp.TORSO_ROLL]

        return tq

def next_target(start, heading, length, env):
    c = env.controller
    rot = utils.rotmatrix(-heading)
    import curriculum as cur
    z_diff = cur.SETTINGS_3D_EASY['z_mean']
    lateral_length = z_diff if c.swing_idx == RIGHT_IDX else -z_diff
    offset = [length, 0.0, lateral_length]
    target = start + np.dot(rot, offset)
    return target

def test(env, length, r=1, n=8, a=0.0, delta_a=0.0, relative=False, mirror=False):
    seed = np.random.randint(100000)
    obs = env.reset(seed=seed)
    if mirror:
        obs.mirror()
    obs.rotate(a)
    env.reset(obs, video_save_dir=None, render=r)
    env.sdf_loader.put_grounds([[-3.0,env.consts().GROUND_LEVEL,0]])
    t = env.controller.stance_heel
    for i in range(n):
        l = length*0.5 if i == 0 else length
        t = next_target(t, a, l, env)
        action = np.zeros(sp.N_PARAMS)
        action[sp.STANCE_YAW] = delta_a # Works approximately, for small delta_a
        if i%2 == 0:
            action[sp.STANCE_YAW] *= -1
        _, terminated = env.simulate(t, action=action, put_dots=True)
        if relative:
            t = env.controller.stance_heel # Pretend that was the previous target
        a += delta_a
        if terminated:
            break

def t_simple():
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv()
    env.sdf_loader.ground_width = 8.0
    test(env, 0.4, delta_a=-.33, n=8)

def t_darwin():
    from darwin_env import DarwinEnv
    env = DarwinEnv()
    test(env, 0.15, r=3, n=60, mirror=True)

if __name__ == "__main__":
    t_simple()
