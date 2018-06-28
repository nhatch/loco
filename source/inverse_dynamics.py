from IPython import embed
import numpy as np
from simbicon import Simbicon
import pickle
import os

from sklearn.linear_model import Ridge

N_ACTIONS_PER_STATE = 4
N_STATES_PER_ITER = 32
MINIBATCH_SIZE = 16 # Actually, maybe stochastic GD is not a good idea?
EXPLORATION_STD = 0.1
START_STATES_FILENAME = 'data/start_states.pkl'
TRAIN_FILENAME = 'data/train.pkl'

RIDGE_ALPHA = 10.0

class LearnInverseDynamics:
    def __init__(self, env):
        self.env = env
        self.initialize_start_states()
        self.train_set = []
        # Use a linear policy for now
        target_space_shape = 2
        self.n_action = env.action_space.shape[0]
        self.n_dynamic = env.observation_space.shape[0] + target_space_shape
        self.lm = Ridge(alpha=RIDGE_ALPHA)
        # Initialize the model with all zeros
        self.lm.fit(np.zeros((1, self.n_dynamic)), np.zeros((1,self.n_action)))

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
        self.train_inverse_dynamics()

    def collect_samples(self, start_state):
        mean_action = self.act(start_state, target=None)
        for _ in range(N_ACTIONS_PER_STATE):
            perturbation = EXPLORATION_STD * np.random.randn(len(mean_action))
            action = mean_action + perturbation
            self.env.reset(start_state)
            end_state, step_dist = self.env.simulate(action)
            if end_state is not None:
                achieved_target = (step_dist, end_state[9]) # Include ending velocity
                self.train_set.append((start_state, action, achieved_target))
                self.start_states.append(end_state)
            else:
                print("ERROR: Ignoring this training datapoint")

    def act(self, state, target=None):
        if target is None:
            target = self.generate_target(state)

        # TODO whiten actions and observations?
        return self.lm.predict(np.concatenate((state, target)).reshape(1,-1)).reshape(-1)

    def generate_target(self, prev_target=None):
        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        if prev_target is not None:
            current_v = prev_target[1]
        else:
            current_v = 1.0
        target_dist = current_v * 0.5 + (0.5 * np.random.uniform() - 0.25)
        target_v = current_v + (0.3 * np.random.uniform() - 0.15)
        target = [target_dist, target_v]
        self.show_target(target)
        return target

    def show_target(self, target):
        absolute_x = self.env.controller.stance_heel + target[0]
        self.env.put_dot(absolute_x, 0.0)

    def collect_dataset(self):
        for i in np.random.choice(range(len(self.start_states)), size=N_STATES_PER_ITER):
            self.collect_samples(self.start_states[i])
        print(len(self.start_states), len(self.train_set))

    def training_iter(self, demo=True):
        print("Starting new training iteration")
        self.collect_dataset()
        self.train_inverse_dynamics()
        if demo:
            self.evaluate(render=1.0)

    def train_inverse_dynamics(self):
        # TODO find some library to do gradient descent
        # Or just implement it myself, it's not that hard...
        X = np.zeros((len(self.train_set), self.n_dynamic))
        y = np.zeros((len(self.train_set), self.n_action))
        for i, sample in enumerate(self.train_set):
            X[i] = np.concatenate((sample[0], sample[2]))
            y[i] = sample[1]
        self.lm.fit(X, y)

    def run_step(self, start_state, render, target):
        action = self.act(start_state, target=target)
        state, step_dist = self.env.simulate(action, render=render)
        if state is None:
            # The robot crashed or something. Just hack so the script doesn't *also* crash.
            state = start_state
            step_dist = -1 # Should make this bigger, but then the graph gets way out of scale
        loss = (step_dist - target[0])**2 + (state[9] - target[1])**2
        error = np.abs(step_dist - target[0])
        return state, loss, error

    def evaluate(self, n_steps=8, render=None, record_video=False):
        state = self.env.reset(record_video=record_video)
        total_loss = 0
        max_error = 0
        t = self.generate_target()
        for _ in range(n_steps):
            state, loss, error = self.run_step(state, render, t)
            if error > max_error:
                max_error = error
            total_loss += loss
            t = self.generate_target(t)
        avg_loss = total_loss/n_steps
        self.env.reset() # This ensures the video recorder is closed properly.
        return avg_loss, max_error

    def demo_train_example(self, i, render=3.0, show_orig_action=False):
        start, action, target = self.train_set[i]
        self.env.reset(start)
        self.show_target(target)
        if not show_orig_action:
            action = self.act(start, target=target)
        state, step_dist = self.env.simulate(action, render=render)
        print("Achieved dist {:.3f} ({:.3f}), vel {:.3f} ({:.3f})".format(step_dist, target[0], state[9], target[1]))

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon)
    learn = LearnInverseDynamics(env)
    #learn.load_train_set()
    #learn.training_iter()
    #learn.training_iter()
    embed()
