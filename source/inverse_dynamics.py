from IPython import embed
import numpy as np
from simbicon import Simbicon

N_ACTIONS_PER_STATE = 4
N_STATES_PER_ITER = 32
MINIBATCH_SIZE = 16 # Actually, maybe stochastic GD is not a good idea?
EXPLORATION_STD = 0.1

def flip_stance(state):
    # Flips the left state with the right state.
    state = state.copy()
    temp = state[3:6]
    state[3:6] = state[6:9]
    state[6:9] = temp
    temp = state[12:15]
    state[12:15] = state[15:18]
    state[15:18] = temp
    return state

class LearnInvDynamics:
    def __init__(self, env):
        self.env = env
        self.initialize_start_states()
        self.train_set = []
        # Use a linear policy for now
        target_space_shape = 2
        self.f = np.zeros((env.action_space.shape[0],
                           env.observation_space.shape[0] + target_space_shape))

    def initialize_start_states(self):
        self.start_states = env.collect_starting_states()

    def collect_samples(self, start_state):
        current_x = start_state[0] # Should this be stance heel location instead?
        current_v = start_state[9]

        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        # TODO should target locations be relative or absolute?
        target_dist = 0.5 + (0.5 * np.random.uniform() - 0.25)
        target_v = current_v + (0.5 * np.random.uniform() - 0.25)
        target = [target_dist, target_v]

        # TODO whiten actions and observations?
        mean_action = np.dot(self.f, np.concatenate((start_state, target)))

        for _ in range(N_ACTIONS_PER_STATE):
            perturbation = EXPLORATION_STD * np.random.randn(len(mean_action))
            action = mean_action + perturbation
            end_state, step_dist = self.env.simulate(start_state, action)
            achieved_target = (step_dist, end_state[9]) # Include ending velocity
            self.train_set.append((start_state, action, achieved_target))
            self.start_states.append(flip_stance(end_state))

    def collect_dataset(self):
        for start_state in np.random.choice(self.start_states, size=N_STATES_PER_ITER):
            self.collect_samples(start_state)

    def training_iter(self):
        self.collect_dataset()
        self.train_inverse_dynamics()

    def train_inverse_dynamics(self):
        # TODO find some library to do gradient descent
        # Or just implement it myself, it's not that hard...
        print(len(self.start_states), len(self.train_set))

    def demo_start_state(self, i, n_steps=12):
        start_state = self.start_states[i]
        action = np.zeros(self.f.shape[0])
        self.env.reset()
        self.env.visual=True
        self.env.simulate(start_state, action)
        for _ in range(n_steps-1):
            self.env.simulate()
        self.env.visual=False

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon, render_factor=3)
    learn = LearnInvDynamics(env)
    #learn.training_iter()
    embed()
