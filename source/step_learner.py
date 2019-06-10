
import numpy as np
from IPython import embed
import cma_wrapper
import curriculum as cur
from evaluator import Evaluator
import simbicon_params as sp
import utils

class Runner:
    def __init__(self, env, start_state, target, use_stepping_stones=True, raw_pose_start=None):
        self.env = env
        self.start_state = start_state
        self.raw_pose_start = raw_pose_start
        self.grounds = start_state.starting_platforms() + [target]
        if not use_stepping_stones:
            self.grounds = self.grounds[:1]
        self.target = target
        self.n_runs = 0

    def run(self, action):
        self.n_runs += 1
        r, _ = self.env.simulate(self.target, target_heading=0.0, action=action, put_dots=True)
        if self.env.video_recorder is not None:
            self.env.pause(0.3)
        return utils.reward(self.env.controller, r)

    def reset(self, video_save_dir=None, render=1):
        self.env.reset(self.start_state, video_save_dir=video_save_dir, render=render)
        self.env.sdf_loader.put_grounds(self.grounds)
        if self.raw_pose_start is not None:
            self.env.robot_skeleton.x = self.raw_pose_start
        if render is not None:
            self.env.render()

def collect_start_state(env, targets, video_save_dir, use_stepping_stones=True):
    env.reset(video_save_dir=video_save_dir, render=1.0)
    if use_stepping_stones:
        env.sdf_loader.put_grounds(targets)
    else:
        env.sdf_loader.put_grounds(targets[:1])
    env.render()
    steps = targets[2:-1] if env.is_3D else targets[1:-1] # TODO refactor so 2D also uses Evaluator
    for t in steps:
        end_state, _ = env.simulate(t, target_heading=0.0, put_dots=True)
        env.render()
    env.clear_skeletons()
    return end_state

def learn_last_move(env, opzer, targets, video_save_dir=None):
    start_state = collect_start_state(env, targets, video_save_dir)
    runner = Runner(env, start_state, targets[-1])
    settings = cur.TRAIN_SETTINGS_3D if env.is_3D else cur.TRAIN_SETTINGS_2D
    opzer.reset()
    return opzer.optimize(runner, np.zeros((sp.N_PARAMS,1)), settings)

def test_2D():
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    env.sdf_loader.ground_length = 0.1
    opzer = cma_wrapper.CMAWrapper()
    LONG_STEP_2D = np.array([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [1.4, 0, 0], [1.7, 0, 0]])
    STAIR_2D = np.array([[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0], [1.2, 0.1, 0]])
    learn_last_move(env, opzer, LONG_STEP_2D)
    env.clear_skeletons()
    learn_last_move(env, opzer, STAIR_2D)

def test_3D(video_save_dir):
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    env = Simple3DEnv(Simbicon3D)
    evaluator = Evaluator(env)
    settings = cur.SETTINGS_3D_EASY
    settings['n_steps'] = 3
    evaluator.set_eval_settings(settings)
    start_state = env.reset()
    BASIC_3D = evaluator.generate_targets(start_state)
    opzer = cma_wrapper.CMAWrapper()
    GY = env.consts().GROUND_LEVEL
    learn_last_move(env, opzer, BASIC_3D[:-1], video_save_dir=video_save_dir)
    env.simulate(BASIC_3D[-1], 0.0)
    #env.clear_skeletons()
    # LONG_STEP is a little too hard, but BASIC at least should be learnable
    LONG_STEP_3D = np.array([[0, GY, 0], [0.3, GY, 0.1], [0.6, GY, -0.1], [1.4, GY, 0.1]])
    #learn_last_move(env, opzer, LONG_STEP_3D, video_save_dir=video_save_dir)
    #env.simulate([1.7, -.9, -.1], 0.0)

if __name__ == '__main__':
    #test_2D()
    test_3D(None)
    embed()
