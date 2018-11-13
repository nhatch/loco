
import numpy as np
from IPython import embed
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

def collect_start_state(env, targets):
    action = np.concatenate(env.controller.base_gait())*0
    # Two short steps, then a long step. This causes issues for the basic SIMBICON
    # controller, if the next step is another short step. This is because the
    # velocity-based swing hip correction tends to overcorrect so that the robot
    # steps far past the next target.
    env.reset(random=0.0)
    env.sdf_loader.put_grounds(targets, runway_length=20)
    env.render()
    for t in targets[1:-1]:
        end_state, _ = env.simulate(t, action=action, render=1, put_dots=True)
        env.render()
    env.clear_skeletons()
    return end_state

def learn_last_move(env, targets):
    start_state = collect_start_state(env, targets)
    runner = Runner(env, start_state, targets[-1])
    rs = RandomSearch(runner, 4, step_size=0.1, eps=0.05)
    rs.random_search()
    env.clear_skeletons()

GY = -0.9 # Ground level for the 3D environment
LONG_STEP_3D = np.array([[0, GY, 0], [0.3, GY, 0.1], [0.6, GY, -0.1], [1.4, GY, 0.1]])
LONG_STEP_2D = np.array([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [1.4, 0, 0], [1.7, 0, 0]])
STAIR_2D = np.array([[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0], [1.2, 0.1, 0]])

if __name__ == '__main__':
    mode = '3D'
    if mode == '2D':
        from stepping_stones_env import SteppingStonesEnv
        env = SteppingStonesEnv()
        learn_last_move(env, LONG_STEP_2D)
        learn_last_move(env, STAIR_2D)
    else:
        from simple_3D_env import Simple3DEnv
        from simbicon_3D import Simbicon3D
        env = Simple3DEnv(Simbicon3D)
        env.sdf_loader.ground_width = 2.0
        learn_last_move(env, LONG_STEP_3D)
    embed()
