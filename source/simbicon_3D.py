import numpy as np
from IPython import embed

from simbicon import Simbicon, UP, DOWN
from sdf_loader import RED, GREEN, BLUE

from consts_common3D import *
from simbicon_params import *

class Simbicon3D(Simbicon):
    # TODO test this
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
        return state

    def base_gait(self):
        gait = [0.14, 0.5, 0.2, -0.1, 0.2,
                0.4, -1.1,   0.0, -0.05,
                -0.0, -0.00, 0.20, -0.1,
                0,0,0,
                0.7, 0.2, 0.0, 0.0, 0.0]
        return gait

    def controllable_indices(self):
        return np.array([1, 1, 1, 1, 1,
                         0, 0, 0, 0,
                         0, 0, 0, 1,
                         0,0,0,
                         0, 0, 1, 1, 1])

    def compute_target_q(self, q, dq):
        tq = super().compute_target_q(q, dq)

        c = self.env.consts()
        params = self.params

        cd = params[POSITION_BALANCE_GAIN_LAT]
        cv = params[VELOCITY_BALANCE_GAIN_LAT]
        proj = q[:3]
        proj[Y] = self.stance_heel[Y]
        #self.env.sdf_loader.put_dot(proj, color=BLUE, index=1)
        #self.env.sdf_loader.put_dot(self.stance_heel, color=GREEN, index=2)
        d = q[Z] - self.stance_heel[Z]
        if d*dq[Z] < 0:
            # If COM is moving (laterally) towards the stance heel, use no velocity gain.
            # This attempts to correct for a "sashay" issue that was causing self collisions.
            cv = 0
        balance_feedback = -(cd*d + cv*dq[Z])

        tq[self.swing_idx+HIP_ROLL] = balance_feedback - q[ROOT_ROLL]
        # TODO is there a principled way to control yaw?
        # For instance, how might we get the robot to walk in a circle?
        tq[self.stance_idx+HIP_ROLL] = params[STANCE_HIP_ROLL_EXTRA] + q[ROOT_ROLL]
        tq[self.stance_idx+HIP_YAW] = params[YAW_WORLD] + q[ROOT_YAW]
        tq[self.stance_idx+ANKLE_ROLL] = params[STANCE_ANKLE_ROLL]

        tq[TORSO_ROLL] = -q[ROOT_ROLL]

        self.update_doppelganger(tq)
        return tq

    def update_doppelganger(self, q):
        if self.env.doppelganger is None:
            return
        c = self.env.consts()
        q = q.copy()
        q[Z] = 0.5
        q[X] = self.env.get_x()[0][X] # lowercase x means "full state"; uppercase is X axis. Sry
        self.env.doppelganger.q = self.env.from_features(q)
        self.env.doppelganger.dq = np.zeros(c.Q_DIM)

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

def test(env):
    seed = np.random.randint(100000)
    env.seed(seed)
    env.reset(random=0.0)
    env.sdf_loader.put_dot([0,0,0])
    env.sdf_loader.put_grounds([[0,-0.9,0]], runway_length=20.0)
    for i in range(20):
        t = 0.2 + 0.5*i# + np.random.uniform(low=-0.2, high=0.2)
        # TODO customize target y for each .skel file?
        env.simulate([t,-0.9,0], render=1, put_dots=True)

if __name__ == "__main__":
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv(Simbicon3D)
    env.sdf_loader.ground_width = 2.0
    test(env)
    #test_standardize_stance(env)
    embed()
