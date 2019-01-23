from evaluator import Evaluator
import simbicon_params as sp
import curriculum as cur
import numpy as np
from IPython import embed

def test(use_3D=False):
    if use_3D:
        from simple_3D_env import Simple3DEnv
        from simbicon_3D import Simbicon3D
        env = Simple3DEnv(Simbicon3D)
    else:
        from stepping_stones_env import SteppingStonesEnv
        env = SteppingStonesEnv()

    evaluator = Evaluator(env)

    if use_3D:
        # See curriculum.py for more examples of settings you can use
        #evaluator.set_eval_settings(cur.SETTINGS_3D_EASY)
        evaluator.set_eval_settings(cur.SETTINGS_3D_HARD)
        #evaluator.set_eval_settings(cur.SETTINGS_3D_HARDER)
        train_settings = cur.TRAIN_SETTINGS_3D
    else:
        evaluator.set_eval_settings(cur.SETTINGS_2D_EASY)
        #evaluator.set_eval_settings(cur.SETTINGS_2D_HARD)
        train_settings = cur.TRAIN_SETTINGS_2D

    # Stub example of a policy
    def policy(state):
        # You can use these masks to decrease the dimensionality of the problem.
        input_mask = train_settings['observable_features']
        output_mask = train_settings['controllable_params']
        N_OUTPUTS = len(output_mask)
        N_INPUTS = len(input_mask)

        dimension_is_correct = (state.shape == (N_INPUTS,))
        # If you want to decrease the dimensionality:
        state = state * input_mask
        print("State has dimension {}: {}".format(N_INPUTS, dimension_is_correct))
        action = np.zeros(N_OUTPUTS)
        # If you want to decrease the dimensionality:
        action = action * output_mask
        return action

    result = evaluator.evaluate(policy)
    # `experience` is an array of (state, action, reward) tuples.
    # Note that `state` also includes 3D coordinates for the three most recent
    # step targets, as well as (for convenience) the location of the robot's
    # stance heel.
    # It has been standardized (mirrored, translated, rotated) to remove extraneous
    # information that does not affect the dynamics.
    experience = result['experience']
    embed()

if __name__ == '__main__':
    test(use_3D=False)
