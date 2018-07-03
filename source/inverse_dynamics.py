from IPython import embed
import numpy as np
from simbicon import Simbicon
import pickle
import os

from sklearn.linear_model import Ridge, RANSACRegressor

N_ACTIONS_PER_STATE = 4
N_STATES_PER_ITER = 128
EXPLORATION_STD = 0.1
START_STATES_FILENAME = 'data/start_states.pkl'
TRAIN_FILENAME = 'data/train.pkl'

RIDGE_ALPHA = 10.0

class LearnInverseDynamics:
    def __init__(self, env):
        self.env = env
        self.initialize_start_states()
        self.train_set = []
        target_space_shape = 1
        self.n_action = env.action_space.shape[0]
        self.n_dynamic = env.observation_space.shape[0] + target_space_shape
        # Use a linear policy for now
        r = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        self.lm = RANSACRegressor(base_estimator=r, residual_threshold=2.0)
        self.X_mean = np.zeros(self.n_dynamic)
        self.y_mean = np.zeros(self.n_action)
        # Maybe try varying these.
        # Increasing X factor increases the penalty for using that feature as a predictor.
        # Increasing y factor increases the penalty for mispredicting that action parameter.
        # Originally I tried scaling by the variance, but that led to very poor performance.
        # It appears that variance in whatever arbitrary units is not a good indicator of
        # importance in fitting a linear model of the dynamics for this environment.
        self.X_scale_factor = np.ones(self.n_dynamic)
        self.y_scale_factor = np.ones(self.n_action)

    def initialize_start_states(self):
        if not os.path.exists(START_STATES_FILENAME):
            states = self.env.collect_starting_states()
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
                achieved_target = [step_dist] # Include ending velocity
                self.train_set.append((start_state, action, achieved_target))
                self.start_states.append(end_state)
            else:
                print("ERROR: Ignoring this training datapoint")

    def act(self, state, target=None):
        if target is None:
            target = [self.generate_targets(1)[0]]
        X = np.concatenate((state, target)).reshape(1,-1)
        X = (X - self.X_mean) / self.X_scale_factor
        if hasattr(self.lm, 'estimator_'):
            rescaled_action = self.lm.predict(X).reshape(-1)
        else:
            rescaled_action = np.zeros(self.n_action)
        return rescaled_action / self.y_scale_factor + self.y_mean

    def generate_targets(self, num_steps, mean=0.42, width=0.3):
        targets = []
        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        for _ in range(num_steps):
            dist = mean + width * (np.random.uniform() - 0.5)
            targets.append(dist)
        return targets

    def show_target(self, target):
        absolute_x = self.env.controller.stance_heel + target[0]
        self.env.put_dot(absolute_x, 0.0)

    def collect_dataset(self):
        for i in np.random.choice(range(len(self.start_states)), size=N_STATES_PER_ITER):
            self.collect_samples(self.start_states[i])
        print(len(self.start_states), len(self.train_set))

    def training_iter(self):
        self.collect_dataset()
        self.train_inverse_dynamics()

    def train_inverse_dynamics(self):
        # TODO try some model more complicated than linear?
        X = np.zeros((len(self.train_set), self.n_dynamic))
        y = np.zeros((len(self.train_set), self.n_action))
        for i, sample in enumerate(self.train_set):
            X[i] = np.concatenate((sample[0], sample[2]))
            y[i] = sample[1]
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean) / self.X_scale_factor
        y = (y - self.y_mean) * self.y_scale_factor
        self.lm.fit(X, y)

    def run_step(self, start_state, render, target):
        action = self.act(start_state, target=target)
        state, step_dist = self.env.simulate(action, render=render)
        if state is None:
            # The robot crashed.
            return None, None
        error = np.abs(step_dist - target[0])
        return state, error

    def evaluate(self, n_steps=16, render=None, record_video=False, seed=None, width=0.3, mean=0.42):
        if seed:
            self.env.seed(seed)
        state = self.env.reset(record_video=record_video)
        total_error = 0
        max_error = 0
        total_score = 0
        DISCOUNT = 0.8
        distance_targets = self.generate_targets(n_steps, mean=mean, width=width)
        total_offset = 0
        num_successful_steps = 0
        for i in range(n_steps):
            target = [sum(distance_targets[:i+1]) - self.env.controller.stance_heel]
            self.show_target(target)
            state, error = self.run_step(state, render, target)
            if state is None:
                # The robot crashed
                break
            num_successful_steps += 1
            if error > max_error:
                max_error = error
            total_error += error
            if error < 1:
                total_score += (1-error) * (DISCOUNT**i)
        self.env.reset() # This ensures the video recorder is closed properly.
        result = {
                "total_score": total_score,
                "n_steps": num_successful_steps,
                "max_error": max_error,
                "total_error": total_error,
                }
        return result

    def demo_train_example(self, i, render=3.0, show_orig_action=False):
        start, action, target = self.train_set[i]
        self.env.reset(start)
        self.show_target(target)
        if not show_orig_action:
            action = self.act(start, target=target)
        state, step_dist = self.env.simulate(action, render=render)
        print("Achieved dist {:.3f} ({:.3f})".format(step_dist, target[0]))

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon)
    learn = LearnInverseDynamics(env)
    #learn.load_train_set()
    #learn.training_iter()
    embed()
