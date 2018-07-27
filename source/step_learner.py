
import numpy as np
from IPython import embed
from simbicon import Simbicon, SIMBICON_ACTION_SIZE
from random_search import RandomSearch

class Runner:
    def __init__(self, env, start_state, target):
        self.env = env
        self.start_state = start_state
        self.grounds = start_state.starting_platforms() + [target]
        self.target = target

    def run(self, action, render=None):
        self.reset()
        r, _ = self.env.simulate(self.target, action=action, put_dots=True, render=render)
        score = -np.linalg.norm(r.stance_heel_location() - self.target)
        return score

    def reset(self):
        self.env.reset(self.start_state, random=0.0)
        self.env.sdf_loader.put_grounds(self.grounds)

def collect_long_step_start_state(env):
    action = np.zeros(SIMBICON_ACTION_SIZE)
    # Two short steps, then a long step. This causes issues for the basic SIMBICON
    # controller, if the next step is another short step. This is because the
    # velocity-based swing hip correction tends to overcorrect so that the robot
    # steps far past the next target.
    env.reset(random=0.0)
    env.sdf_loader.put_grounds([[0,0]], runway_length=20)
    targets = [0.3, 0.6, 1.4]
    for t in targets:
        end_state, _ = env.simulate([t,0], action=action, render=1, put_dots=True)
    env.clear_skeletons()
    return end_state

def learn_long_step(env):
    start_state = collect_long_step_start_state(env)
    runner = Runner(env, start_state, [1.7, 0.0])
    rs = RandomSearch(env, runner, 4, 0.1, 0.05)
    rs.random_search()
    embed()

def collect_stair_start_state(env):
    action = np.zeros(SIMBICON_ACTION_SIZE)
    env.reset(random=0.0)
    targets = np.array([[0, 0], [0.4, 0], [0.8, 0], [1.2, 0.1]])
    env.sdf_loader.put_grounds(targets)
    for t in range(2):
        target = [(1+t)*0.4, 0]
        end_state, _ = env.simulate(target, action=action, render=1, put_dots=True)
    env.clear_skeletons()
    return end_state

def learn_stair(env):
    start_state = collect_stair_start_state(env)
    runner = Runner(env, start_state, [1.2, 0.1])
    rs = RandomSearch(env, runner, 4, 0.1, 0.05)
    rs.random_search()
    embed()

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    learn_long_step(env)
    #learn_stair(env)
