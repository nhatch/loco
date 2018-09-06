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
        # The following "works" but isn't fast or natural-looking, and it's hard to
        # adjust to get it to work for different step lengths.
        #gait = ([0.14, 0.5, 0.2, -0.2,  0.5, -1.1,   0.6, -0.05,   0, 0.5, 0.2],
        #        [0.14, 0.5, 0.2, -0.2, -0.1, -0.05, 0.15, -0.1, 0.0, 0.5, 0.2])
        # The following are the settings used for the 2D model. They don't work well in 3D.
        # (Yaw is poorly controlled, and the swing foot collides with the stance foot.)
        gait = ([0.14, 0.5, 0.2, -0.05,  0.4, -1.1,   0.0, -0.05,0.2, 0.5, 0.2],
                [0.14, 0.5, 0.2, -0.05, -0.0, -0.00, 0.20, -0.1, 0.2, 0.5, 0.2])
        return gait

    def adjust_up_targets(self):
        pass

    def maybe_start_down_phase(self, contacts, swing_heel):
        duration = self.time() - self.step_started
        if duration > 0.3:
            self.direction = DOWN

    def compute_target_q(self, q, dq):
        tq = super().compute_target_q(q, dq)

        c = self.env.consts()
        state = self.FSMstate()

        cd = state.position_balance_gain_lat
        cv = state.velocity_balance_gain_lat
        proj = q[:3]
        proj[1] = self.stance_heel[1]
        self.env.sdf_loader.put_dot(proj, color=BLUE, index=1)
        self.env.sdf_loader.put_dot(self.stance_heel, color=GREEN, index=2)
        d = q[2] - self.stance_heel[2]
        balance_feedback = -(cd*d + cv*dq[2])

        tq[self.swing_idx+2] = balance_feedback - q[c.PITCH_IDX+2]
        # TODO there might be a more principled way to control the extra DOFs than
        # the following. But this works for now.
        tq[self.stance_idx+2] = q[c.PITCH_IDX+2]
        tq[self.stance_idx+1] = 2*q[c.PITCH_IDX+1]
        tq[18] = -q[c.PITCH_IDX+2]

        #actual_twist = self.skel.q[4] + self.skel.q[7]
        # TODO why does the robot rotate so much on the third step of test()?
        # Try putting dots on all the contact points -- does the robot lose contact
        # with the stance foot (so it's a conservation of angular momentum thing)?
        #q[self.stance_idx+c.HIP_OFFSET_TWIST] = actual_twist/2
        #q[self.swing_idx+c.HIP_OFFSET_TWIST] = -actual_twist/2

        self.update_doppelganger(tq)
        return tq

    def update_doppelganger(self, q):
        if self.env.doppelganger is None:
            return
        c = self.env.consts()
        q = q.copy()
        q[2] = 0.5
        q[0] = self.env.get_x()[0][0]
        self.env.doppelganger.q = self.env.from_features(q)
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
    for i in range(20):
        t = 0.1 + 0.2*i# + np.random.uniform(low=-0.2, high=0.2)
        # TODO customize target y for each .skel file?
        env.simulate([t,-0.9,0], render=1, put_dots=False)

if __name__ == "__main__":
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv(Simbicon3D)
    test(env)
    embed()
