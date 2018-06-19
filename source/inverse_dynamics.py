from IPython import embed
import numpy as np
from simbicon import Simbicon
import pickle
import os

N_ACTIONS_PER_STATE = 4
N_STATES_PER_ITER = 32
MINIBATCH_SIZE = 16 # Actually, maybe stochastic GD is not a good idea?
EXPLORATION_STD = 0.1
START_STATES_FILENAME = 'data/start_states.pkl'
TRAIN_FILENAME = 'data/train.pkl'

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
        if not os.path.exists(START_STATES_FILENAME):
            states = env.collect_starting_states()
            with open(START_STATES_FILENAME, 'wb') as f:
                pickle.dump(states, f)
        with open(START_STATES_FILENAME, 'rb') as f:
            self.start_states = pickle.load(f)

    def dump_train_set(self):
        with open(TRAIN_FILENAME, 'wb') as f:
            pickle.dump((self.start_states, self.train_set), f)

    def load_train_set(self):
        with open(TRAIN_FILENAME, 'rb') as f:
            self.start_states, self.train_set = pickle.load(f)

    def collect_samples(self, start_state):
        mean_action = self.act(start_state, target=None)
        for _ in range(N_ACTIONS_PER_STATE):
            perturbation = EXPLORATION_STD * np.random.randn(len(mean_action))
            action = mean_action + perturbation
            self.env.reset()
            end_state, step_dist = self.env.simulate(start_state, action)
            if end_state is not None:
                achieved_target = (step_dist, end_state[9]) # Include ending velocity
                self.train_set.append((start_state, action, achieved_target))
                self.start_states.append(end_state)
            else:
                print("ERROR: Ignoring this training datapoint")

    def act(self, state, target=None):
        if target is None:
            # Choose some random targets that are hopefully representative of what
            # we will see during testing episodes.
            current_v = state[9]
            target_dist = 0.5 + (0.5 * np.random.uniform() - 0.25)
            target_v = current_v + (0.5 * np.random.uniform() - 0.25)
            target = [target_dist, target_v]

        # TODO whiten actions and observations?
        return np.dot(self.f, np.concatenate((state, target)))

    def collect_dataset(self):
        for i in np.random.choice(range(len(self.start_states)), size=N_STATES_PER_ITER):
            self.collect_samples(self.start_states[i])

    def training_iter(self):
        print("Starting new training iteration")
        self.collect_dataset()
        self.train_inverse_dynamics()

    def train_inverse_dynamics(self):
        # TODO find some library to do gradient descent
        # Or just implement it myself, it's not that hard...
        print(len(self.start_states), len(self.train_set))

    def demo_start_state(self, i, n_steps=12, render=1.0):
        start_state = self.start_states[i]
        self.env.reset()
        action = self.act(start_state)
        state, _ = self.env.simulate(start_state, action, render=render)
        for _ in range(n_steps-1):
            action = self.act(state)
            state, _ = self.env.simulate(action=action, render=render)

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon)
    learn = LearnInvDynamics(env)
    learn.load_train_set()
    #learn.training_iter()
    embed()