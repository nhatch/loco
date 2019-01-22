import numpy as np
from IPython import embed
from inverse_dynamics import LearnInverseDynamics
from step_learner import Runner
from state import reconstruct_state

TRAIN_SETTINGS_3D_TEST = {
    'n_trajectories': 1,
    'n_dirs': 8,
    'tol': 0.05,
    }

SETTINGS_3D_TEST = {
    'use_stepping_stones': False,
    'dist_mean': 0.35,
    'dist_spread': 0.0,
    'max_intolerable_steps': 2,
    'early_termination_tol': 0.05,
    'n_steps': 16,
    'z_mean': 0.4,
    'z_spread': 0.0,
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
    start_state, target = reconstruct_state(features, learn.env.consts())
    return start_state, target, response, features

def demo_train_set(learn):
    for i in range(len(learn.train_responses)):
        start_state, target, response, features = retrieve_index(learn, i)
        runner = Runner(learn.env, start_state, target)
        runner.reset(render=1.0)
        print("Score:", runner.run(response)) # Should be pretty close to zero.

def test_regression_bias(learn, i=None):
    start_state, target, response, features = retrieve_index(learn, i)

    runner = Runner(learn.env, start_state, target)
    runner.reset(render=1.0)
    print("Score:", runner.run(response)) # Should be pretty close to zero.

    runner.reset(render=1.0)
    trained_response = learn.act(features)
    # This score should also be close to zero, depending on model bias.
    print("Score:", runner.run(trained_response))

    # TODO: It seems like the algorithm is almost (not completely) ignoring the target Z coord.
    target[2] += 0.2
    runner = Runner(learn.env, start_state, target)
    runner.reset(render=1.0)
    trained_response = learn.act(features)
    # This score should also be close to zero in the absence of model bias.
    print("Score:", runner.run(trained_response))

def test_mirroring(learn, i=3):
    start_state, target, response, features = retrieve_index(learn, i)
    runner = Runner(learn.env, start_state, target)
    trained_response = learn.act(features)

    # Note this will only give accurate results if `learn` has been trained
    # on more than one datapoint (say 5 or 6 to be safe).
    mirrored_start = start_state.copy()
    mirrored_start.mirror()
    mirrored_target = target*[1,1,-1]
    mirrored_runner = Runner(learn.env, mirrored_start, mirrored_target)
    mirrored_features = mirrored_start.extract_features(mirrored_target)
    mirrored_response = learn.act(mirrored_features)

    runner.reset(render=1.0)
    score = runner.run(trained_response)
    end_state = env.current_observation()
    mirrored_runner.reset(render=1.0)
    mirrored_score = mirrored_runner.run(mirrored_response)
    mirrored_end = env.current_observation()
    mirrored_end.mirror()

    runner.reset(render=1.0)
    expert_score = runner.run(response)
    mirrored_runner.reset(render=1.0)
    mirrored_expert_score = mirrored_runner.run(response)

    print("Mirrored response:", np.allclose(trained_response, mirrored_response))
    print("Score:            ", np.allclose(score, mirrored_score))
    print("End state:        ", np.allclose(end_state.raw_state, mirrored_end.raw_state, atol=1e-5))
    embed()
    print("Expert score:     ", np.allclose(expert_score, mirrored_expert_score, atol=1e-5))

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    #env = SteppingStonesEnv()
    env = Simple3DEnv(Simbicon3D)

    name = 'test'
    learn = LearnInverseDynamics(env, name)
    #learn.load_train_set()
    learn.evaluator.set_eval_settings(SETTINGS_3D_TEST)
    learn.set_train_settings(TRAIN_SETTINGS_3D_TEST)
    #print('Score:', learn.evaluate()['total_score'])
    for i in range(2):
        learn.training_iter()
    #test_regression_bias(learn)
    test_mirroring(learn)
    embed()
