
import numpy as np
from IPython import embed
from random_search import RandomSearch
from curriculum import *

class Runner:
    def __init__(self, env, start_state, target, use_stepping_stones=True):
        self.env = env
        self.start_state = start_state
        self.grounds = start_state.starting_platforms() + [target]
        if not use_stepping_stones:
            self.grounds = self.grounds[:1]
        self.target = target

    def run(self, action):
        r, _ = self.env.simulate(self.target, target_heading=0.0, action=action, put_dots=True)
        score = -np.linalg.norm(r.stance_heel_location() - r.stance_platform())
        if self.env.video_recorder is not None:
            self.env.pause(0.3)
        return score

    def reset(self, video_save_dir=None, render=None):
        self.env.reset(self.start_state, video_save_dir=video_save_dir, render=render)
        self.env.sdf_loader.put_grounds(self.grounds)

def collect_start_state(env, targets, video_save_dir, use_stepping_stones=True):
    # Two short steps, then a long step. This causes issues for the basic SIMBICON
    # controller, if the next step is another short step. This is because the
    # velocity-based swing hip correction tends to overcorrect so that the robot
    # steps far past the next target.
    env.reset(video_save_dir=video_save_dir, render=1.0)
    if use_stepping_stones:
        env.sdf_loader.put_grounds(targets)
    else:
        env.sdf_loader.put_grounds(targets[:1])
    env.render()
    for t in targets[1:-1]:
        end_state, _ = env.simulate(t, target_heading=0.0, put_dots=True)
        env.render()
    env.clear_skeletons()
    return end_state

def learn_last_move(env, targets, video_save_dir=None):
    stones = not env.is_3D
    start_state = collect_start_state(env, targets, video_save_dir, use_stepping_stones=stones)
    runner = Runner(env, start_state, targets[-1], use_stepping_stones=stones)
    if env.is_3D:
        rs = RandomSearch(runner, TRAIN_SETTINGS_3D)
    else:
        rs = RandomSearch(runner, TRAIN_SETTINGS_2D)
    rs.random_search(render=1, video_save_dir=video_save_dir)
    return rs

def test_2D():
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    env.sdf_loader.ground_length = 0.1
    LONG_STEP_2D = np.array([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [1.4, 0, 0], [1.7, 0, 0]])
    STAIR_2D = np.array([[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0], [1.2, 0.1, 0]])
    learn_last_move(env, LONG_STEP_2D)
    env.clear_skeletons()
    return learn_last_move(env, STAIR_2D)

def test_3D(video_save_dir):
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    env = Simple3DEnv(Simbicon3D)
    GY = -0.9 # Ground level for the 3D environment
    # LONG_STEP is a little too hard, but BASIC at least should be learnable
    LONG_STEP_3D = np.array([[0, GY, 0], [0.3, GY, 0.1], [0.6, GY, -0.1], [1.4, GY, 0.1]])
    BASIC_3D = np.array([[0, GY, 0], [0.2, GY, 0.1], [0.7, GY, -0.1], [1.2, GY, 0.1]])
    rs = learn_last_move(env, BASIC_3D, video_save_dir=video_save_dir)
    env.simulate([1.7, -.9, -.1], 0.3)
    env.clear_skeletons()
    rs = learn_last_move(env, LONG_STEP_3D, video_save_dir=video_save_dir)
    env.simulate([1.7, -.9, -.1], 0.3)
    return rs

if __name__ == '__main__':
    #rs = test_2D()
    rs = test_3D(None)
    embed()
