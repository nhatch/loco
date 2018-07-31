import numpy as np
from IPython import embed

from simbicon import Simbicon, UP, DOWN
from sdf_loader import RED, GREEN, BLUE

class Simbicon3D(Simbicon):
    # TODO test this
    def standardize_stance(self, state):
        c = self.env.consts()
        state = super().standardize_stance(state)
        # Rotations are absolute, not relative, so we need to multiply some angles
        # by -1 to obtain a mirrored pose.
        D = c.Q_DIM
        m = np.ones(D)
        absolute_rotation_indices = [2,4,5,6,7,10,11,14,16,17,20]
        m[absolute_rotation_indices] = -1
        state[0:D] *= m
        state[D:2*D] *= m
        return state

    def base_gait(self):
        return ([0.14, 0.5, 0.2, -0.2, 0.5, -1.1, 0.6, -0.05, 0,   0.5, 0.2],
                [0.14, 0.5, 0.2, -0.2, -0.1, -0.05, 0.15, -0.1, 0, 0.5, 0.2])

    def adjust_up_targets(self):
        pass

    def maybe_start_down_phase(self, contacts, swing_heel):
        duration = self.time() - self.step_started
        if duration > 0.3:
            self.direction = DOWN

    def compute_target_q(self):
        q = super().compute_target_q()

        c = self.env.consts()
        state = self.FSMstate()

        cd = state.position_balance_gain_lat
        cv = state.velocity_balance_gain_lat
        v = self.skel.dq[2]
        self.env.sdf_loader.put_dot(self.skel.q[:3], color=BLUE, index=1)
        self.env.sdf_loader.put_dot(self.stance_heel, color=GREEN, index=2)
        d = self.skel.q[2] - self.stance_heel[2]
        balance_feedback = -(cd*d + cv*v)

        base = -1 if self.swing_idx == c.RIGHT_IDX else 1
        # TODO can I get rid of this base separation by better initializing
        # the stance_heel location? Now that we're in 3D, the lateral location
        # is important, so just setting it to (0,0,0) doesn't work well.
        base *= 0.1
        angle = base + balance_feedback
        q[self.swing_idx+c.HIP_OFFSET_LAT] = angle

        self.update_doppelganger(q)
        return q

    def update_doppelganger(self, q):
        c = self.env.consts()
        q = q.copy()
        q[:c.BRICK_DOF] = self.env.robot_skeleton.q[:c.BRICK_DOF]
        q[2] -= 0.5
        self.env.doppelganger.q = q
        self.env.doppelganger.dq = np.zeros(c.Q_DIM)

def test_standardize_stance(env):
    from time import sleep
    env.reset(random=0.5)
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
    for i in range(8):
        t = 0.3 + 0.4*i# + np.random.uniform(low=-0.2, high=0.2)
        env.simulate([t,0,0], render=1, put_dots=False)

if __name__ == "__main__":
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv(Simbicon3D)
    test(env)
    embed()
