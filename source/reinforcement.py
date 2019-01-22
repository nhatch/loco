from evaluator import Evaluator
from simbicon_params import PARAM_SCALE
import numpy as np
from IPython import embed
from curriculum import *

def test(use_3D=False):
    if use_3D:
        from simple_3D_env import Simple3DEnv
        from simbicon_3D import Simbicon3D
        env = Simple3DEnv(Simbicon3D)
    else:
        from stepping_stones_env import SteppingStonesEnv
        env = SteppingStonesEnv()

    evaluator = Evaluator(env)
    N_OUTPUTS = len(PARAM_SCALE)
    N_INPUTS = evaluator.input_dimension()

    if use_3D:
        # See curriculum.py for more examples of settings you can use
        #evaluator.set_eval_settings(SETTINGS_3D_EASY)
        evaluator.set_eval_settings(SETTINGS_3D_HARD)
        #evaluator.set_eval_settings(SETTINGS_3D_HARDER)
    else:
        evaluator.set_eval_settings(SETTINGS_2D_EASY)
        #evaluator.set_eval_settings(SETTINGS_2D_HARD)

    # Stub example of a policy
    def policy(state):
        dimension_is_correct = (state.shape == (N_INPUTS,))
        print("State has dimension {}: {}".format(N_INPUTS, dimension_is_correct))
        return np.zeros(N_OUTPUTS)

    result = evaluator.evaluate(policy)
    # This is an array of (state, action, reward) tuples.
    # In RL terms, (state,target) is the environment state.
    experience = result['experience']
    embed()

if __name__ == '__main__':
    test(use_3D=True)
