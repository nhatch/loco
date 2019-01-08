
import numpy as np
from IPython import embed
from random_search import RandomSearch

class Runner:
    def __init__(self, env, start_state, target):
        self.env = env
        self.start_state = start_state
        self.grounds = start_state.starting_platforms() + [target]
        self.target = target
        self.video_save_dir = None

    def run(self, action, render=None):
        video_save_dir = self.video_save_dir if render is not None else None
        self.reset(video_save_dir)
        r, _ = self.env.simulate(self.target, action=action, put_dots=True, render=render)
        score = -np.linalg.norm(r.stance_heel_location() - r.stance_platform())
        if render is not None:
            print("SCORE:", score)
            self.env.pause(0.3)
        return score

    def reset(self, video_save_dir=None):
        self.env.reset(self.start_state, video_save_dir=video_save_dir)
        self.env.sdf_loader.put_grounds(self.grounds)

def collect_start_state(env, targets):
    action = env.controller.base_gait()*0
    # Two short steps, then a long step. This causes issues for the basic SIMBICON
    # controller, if the next step is another short step. This is because the
    # velocity-based swing hip correction tends to overcorrect so that the robot
    # steps far past the next target.
    env.reset()
    env.sdf_loader.put_grounds(targets, runway_length=20)
    env.render()
    for t in targets[1:-1]:
        end_state, _ = env.simulate(t, action=action, render=1, put_dots=True)
        env.render()
    env.clear_skeletons()
    return end_state

def learn_last_move(env, targets):
    start_state = collect_start_state(env, targets)
    # TODO perhaps we should flip the sign on the Z coordinate of the target
    # when the state has been flipped (i.e. standardized).
    # However, for now this is OK because the tests are designed so that the
    # step of interest is *not* flipped (to make them easier to evaluate visually).
    runner = Runner(env, start_state, targets[-1])
    rs = RandomSearch(runner, 4, step_size=0.1, eps=0.05)
    rs.random_search()
    env.clear_skeletons()

def test_2D():
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    LONG_STEP_2D = np.array([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0], [1.4, 0, 0], [1.7, 0, 0]])
    STAIR_2D = np.array([[0, 0, 0], [0.4, 0, 0], [0.8, 0, 0], [1.2, 0.1, 0]])
    learn_last_move(env, LONG_STEP_2D)
    learn_last_move(env, STAIR_2D)

def test_3D():
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    env = Simple3DEnv(Simbicon3D)
    GY = -0.9 # Ground level for the 3D environment
    # LONG_STEP is a little too hard, but BASIC at least should be learnable
    LONG_STEP_3D = np.array([[0, GY, 0], [0.3, GY, 0.1], [0.6, GY, -0.1], [1.4, GY, 0.1]])
    BASIC_3D = np.array([[0, GY, 0], [0.2, GY, 0.1], [0.7, GY, -0.1], [1.2, GY, 0.1]])
    learn_last_move(env, BASIC_3D)
    #learn_last_move(env, LONG_STEP_3D)

if __name__ == '__main__':
    #test_2D()
    test_3D()
    embed()
