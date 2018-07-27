import numpy as np
from IPython import embed

from simbicon import Simbicon, UP, DOWN

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


    def compute_target_q(self):
        q = super().compute_target_q()

        c = self.env.consts()
        state = self.FSMstate()

        cd = state.position_balance_gain_lat
        cv = state.velocity_balance_gain_lat
        v = self.skel.dq[2]
        d = self.skel.q[2] - self.contact[2]
        balance_feedback = cd*d + cv*v

        q[self.swing_idx+c.HIP_OFFSET_LAT] = balance_feedback

        return q

def test_standardize_stance(env):
    from time import sleep
    env.reset(random=0.5)
    c = env.controller
    c.change_stance([], [0,0])
    obs = env.current_observation()
    env.render()
    sleep(0.5)
    env.reset(obs, random=0.0)
    env.render()

if __name__ == "__main__":
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv(Simbicon3D)
    test_standardize_stance(env)
    embed()
