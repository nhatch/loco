
import numpy as np
from IPython import embed
from simbicon import Simbicon, SIMBICON_ACTION_SIZE
import pickle
from random_search import RandomSearch

START_FILENAME = 'data/step_learner_test.pkl'

class Runner:
    def __init__(self, env, start_state):
        self.env = env
        self.start_state = start_state

    def run(self, action, seed, render=None):
        self.env.reset(self.start_state)

        target_x = 0.3
        self.env.simulate(target_x, action=action[:,0], put_dots=True, render=render)
        c = self.env.controller
        score = -np.abs(c.stance_heel - target_x)
        return score

def collect_start_state(env):
    action = np.zeros(SIMBICON_ACTION_SIZE)
    # Two short steps, then a long step. This causes issues for the basic SIMBICON
    # controller, if the next step is another short step. This is because the
    # velocity-based swing hip correction tends to overcorrect so that the robot
    # steps far past the next target.
    targets = [0.3, 0.6, 1.4]
    env.reset(random=0.0)
    for t in targets:
        # TODO: for very small target steps (e.g. 10 cm), the velocity is so small that
        # the robot can get stuck in the UP state, balancing on one leg.
        end_state, _ = env.simulate(t, action=action, render=1, show_dots=True)
        if t == 1.4:
            with open(START_FILENAME, 'wb') as f:
                pickle.dump(end_state, f)

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon)
    env.put_grounds([], runway_length=20)
    with open(START_FILENAME, 'rb') as f:
        start_state = pickle.load(f)
    runner = Runner(env, start_state)
    env.observation_space = np.zeros(1)
    rs = RandomSearch(env, runner, 4, 0.1, 0.05)
    rs.random_search(10)
    embed()


