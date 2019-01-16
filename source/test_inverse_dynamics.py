import numpy as np
from IPython import embed
from inverse_dynamics import LearnInverseDynamics
from step_learner import Runner
from state import State

TRAIN_SETTINGS_2D = {
    'n_dirs': 4,
    'tol': 0.02,
    'step_size': 0.1,
    }

SETTINGS_2D = {
    'use_stepping_stones': True,
    'dist_mean': 0.47,
    'dist_spread': 0.3,
    'ground_length': 0.1,
    'ground_width': 0.5,
    'n_steps': 16,
    'z_mean': 0.0,
    'z_spread': 0.0,
    'y_mean': 0.05,
    'y_spread': 0.1,
    }

TRAIN_SETTINGS_3D = {
    'n_dirs': 8,
    'tol': 0.05,
    'step_size': 0.3,
    }

SETTINGS_3D_EASY = {
    'use_stepping_stones': False,
    'dist_mean': 0.35,
    'dist_spread': 0.0,
    'ground_length': 10.0,
    'ground_width': 2.0,
    'n_steps': 16,
    'z_mean': 0.4,
    'z_spread': 0.0,
    'y_mean': 0.0,
    'y_spread': 0.0,
    }

SETTINGS_3D_MEDIUM = {
    'use_stepping_stones': False,
    'dist_mean': 0.35,
    'dist_spread': 0.2,
    'ground_length': 10.0,
    'ground_width': 2.0,
    'n_steps': 16,
    'z_mean': 0.4,
    'z_spread': 0.1,
    'y_mean': 0.0,
    'y_spread': 0.0,
    }

SETTINGS_3D_HARD = {
    'use_stepping_stones': True,
    'dist_mean': 0.35,
    'dist_spread': 0.5,
    'ground_length': 0.3,
    'ground_width': 0.2,
    'n_steps': 16,
    'z_mean': 0.4,
    'z_spread': 0.2,
    'y_mean': 0.0,
    'y_spread': 0.0,
    }

def retrieve_index(learn, i=None):
    learn.env.clear_skeletons()
    if i is None:
        i = np.random.randint(len(learn.train_features))
        print("Testing index:", i)
    # TODO: restore the evaluation settings that were used at this training index.
    features = learn.train_features[i]
    response = learn.train_responses[i]
    start_state, target = learn.reconstruct_state(features)
    return start_state, target, response

def demo_train_set(learn):
    for i in range(len(learn.train_responses)):
        start_state, target, response = retrieve_index(learn, i)
        runner = Runner(learn.env, start_state, target)
        runner.reset(render=1.0)
        print("Score:", runner.run(response)) # Should be pretty close to zero.

def test_regression_bias(learn, i=None):
    start_state, target, response = retrieve_index(learn, i)

    runner = Runner(learn.env, start_state, target)
    runner.reset(render=1.0)
    print("Score:", runner.run(response)) # Should be pretty close to zero.

    runner.reset(render=1.0)
    trained_response = learn.act(target)
    # This score should also be close to zero, depending on model bias.
    print("Score:", runner.run(trained_response))

    # TODO: It seems like the algorithm is almost (not completely) ignoring the target Z coord.
    target[2] += 0.2
    runner = Runner(learn.env, start_state, target)
    runner.reset(render=1.0)
    trained_response = learn.act(target)
    # This score should also be close to zero in the absence of model bias.
    print("Score:", runner.run(trained_response))

def test_mirroring(learn, i=None):
    start_state, target, response = retrieve_index(learn, i)
    runner = Runner(learn.env, start_state, target)
    trained_response = learn.act(target)
    runner.reset(render=1.0)
    print("Score:", runner.run(trained_response)) # Should be pretty close to zero.

    mirrored_start = State(learn.env.controller.mirror_state(start_state.raw_state.copy()))
    mirrored_target = target*[1,1,-1]
    mirrored_runner = Runner(learn.env, mirrored_start, mirrored_target)
    mirrored_runner.reset(render=1.0)
    learn.env.controller.change_stance([], mirrored_start.stance_heel_location())
    obs_after_mirror = learn.env.current_observation()
    response_after_mirror = learn.act(target)
    print(np.allclose(obs_after_mirror.raw_state, start_state.raw_state))
    print(np.allclose(response_after_mirror, trained_response))
    # This score should be identical to the previous score (actually about 1e-6 error)
    print("Score:", mirrored_runner.run(trained_response))

    # These scores should also be identical (actually about 1e-6 error)
    runner.reset(render=1.0)
    print("Score:", runner.run(response))
    mirrored_runner.reset(render=1.0)
    learn.env.controller.change_stance([], mirrored_start.stance_heel_location())
    print("Score:", mirrored_runner.run(response))

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    #env = SteppingStonesEnv()
    env = Simple3DEnv(Simbicon3D)

    name = 'properdagger'
    learn = LearnInverseDynamics(env, name)
    learn.set_eval_settings(SETTINGS_3D_EASY)
    learn.set_train_settings(TRAIN_SETTINGS_3D)
    for i in range(3):
        learn.training_iter()
    learn.set_eval_settings(SETTINGS_3D_MEDIUM)
    for i in range(3):
        learn.training_iter()
    #learn.load_train_set()
    #print('Score:', learn.evaluate()['total_score'])
    #test_regression_bias(learn)
    #test_mirroring(learn)
    embed()
