from IPython import embed
import numpy as np
from simbicon import Simbicon
import pickle
import os

from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

N_ACTIONS_PER_STATE = 4
N_STATES_PER_ITER = 128
EXPLORATION_STD = 0.1
START_STATES_FILENAME = 'data/start_states.pkl'
TRAIN_FILENAME = 'data/train.pkl'

RIDGE_ALPHA = 10.0


settings = {
    "dist_mean": 0.42,
    "dist_spread": 0.3,
    "runway_length": 0.4,
    "ground_length": 0.1,
    "n_steps": 16
    }

controllable_indices = [0, 1, 1, 1, 0, 0, 0, 0, 0,
                        1, 0, 0, 1, 0, 0, 0, 1, 1]

class LearnInverseDynamics:
    def __init__(self, env):
        self.env = env
        self.initialize_start_states()
        self.train_states, self.train_targets = [], []
        target_space_shape = 1
        self.n_action = env.action_space.shape[0]
        # The actual target is sort of part of the state, hence the *2.
        self.n_dynamic = 26
        # Use a linear policy for now
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        #model = make_pipeline(PolynomialFeatures(2), model) # quadratic
        self.model = RANSACRegressor(base_estimator=model, residual_threshold=2.0)
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
            states = self.collect_starting_states()
            with open(START_STATES_FILENAME, 'wb') as f:
                pickle.dump(states, f)
        with open(START_STATES_FILENAME, 'rb') as f:
            self.start_states = pickle.load(f)

    def collect_starting_states(self, size=8, n_resets=16, min_length=0.2, max_length=0.8):
        self.env.log("Collecting initial starting states")
        start_states = []
        self.env.controller.set_gait_raw(np.zeros(self.env.action_space.shape[0]))
        starter = 0.3
        self.env.put_grounds([[0,0]], runway_length=100)
        for i in range(n_resets):
            length = min_length + (max_length - min_length) * (i / n_resets)
            self.env.log("Starting trajectory {}".format(i))
            self.env.reset()
            # TODO should we include this first state? It will be very different from the rest.
            #start_states.append(self.env.robot_skeleton.x)
            target = np.array([0,0])
            for j in range(size):
                prev_target, target = target, np.array([starter + length * j, 0])
                end_state = self.env.simulate(target, render=0.1)
                # We need to collect the location of the previous target in order to
                # place stepping stones properly when resetting to this state.
                start_states.append(end_state)
        self.env.clear_skeletons()
        return start_states

    def dump_train_set(self):
        with open(TRAIN_FILENAME, 'wb') as f:
            pickle.dump((self.start_states, self.train_states, self.train_targets), f)

    def load_train_set(self):
        with open(TRAIN_FILENAME, 'rb') as f:
            self.start_states, self.train_states, self.train_targets = pickle.load(f)
        self.train_inverse_dynamics()

    def collect_samples(self, start_state):
        for _ in range(N_ACTIONS_PER_STATE):
            self.env.reset(start_state)
            target = self.generate_targets(start_state, 1, runway_length=settings["ground_length"])[0]
            mean_action = self.act(start_state, target)
            perturbation = EXPLORATION_STD * np.random.randn(len(mean_action))
            perturbation *= controllable_indices
            action = mean_action + perturbation
            end_state = self.env.simulate(target, action)
            if type(end_state) == str:
                print(end_state, "(Ignoring this training datapoint.)")
            else:
                self.append_to_train_set(start_state, action, end_state)
                self.start_states.append(end_state)

    def append_to_train_set(self, start_state, action, end_state):
        train_state = self.center_state(start_state, end_state[22:24], end_state[20:22])
        self.train_states.append(train_state)
        self.train_targets.append(action)

    def center_state(self, state, target, achieved_target):
        centered_state = np.zeros(self.n_dynamic)

        # We don't include 22:24 since that is where we will set the origin,
        # and we don't include 24:26 because it seems less important and we
        # are desperate to reduce problem dimension.
        centered_state[:22] = state[:22]

        centered_state[22:24] = achieved_target
        centered_state[24:26] = target

        # Absolute location does not affect dynamics, so recenter all (x,y) coordinates
        # around the location of the starting platform.
        base = state[22:24] # starting platform location
        centered_state[ 0: 2] -= base # starting location of agent
        centered_state[18:20] -= base # starting stance contact location
        centered_state[20:22] -= base # starting stance heel location
        centered_state[22:24] -= base # ending heeldown location
        centered_state[24:26] -= base # ending platform location

        return centered_state

    def train_inverse_dynamics(self):
        # TODO try some model more complicated than linear?
        X = np.array(self.train_states)
        y = np.array(self.train_targets)
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean) / self.X_scale_factor
        y = (y - self.y_mean) * self.y_scale_factor
        self.model.fit(X, y)

    def act(self, state, target):
        X = self.center_state(state, target, target).reshape(1,-1)
        X = (X - self.X_mean) / self.X_scale_factor
        if hasattr(self.model, 'estimator_'):
            rescaled_action = self.model.predict(X).reshape(-1)
        else:
            rescaled_action = np.zeros(self.n_action)
        rescaled_action *= controllable_indices
        return rescaled_action / self.y_scale_factor + self.y_mean

    def generate_targets(self, start_state, num_steps, runway_length=3.0):
        # Generate starting platforms for both feet
        targets = [start_state[24:26], start_state[18:20]]
        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        next_target = start_state[18:20]
        for _ in range(num_steps):
            dx = settings["dist_mean"] + settings["dist_spread"] * (np.random.uniform() - 0.5)
            next_target = next_target + [dx, 0]
            targets.append(next_target)
        self.env.put_grounds(targets, ground_length=settings["ground_length"],
                runway_length=runway_length)
        return targets[2:] # Don't include the starting platforms

    def collect_dataset(self):
        self.env.clear_skeletons()
        indices = np.random.choice(range(len(self.start_states)), size=N_STATES_PER_ITER)
        for i,j in enumerate(indices):
            print("Exploring start state {} ({} / {})".format(j, i, len(indices)))
            self.collect_samples(self.start_states[j])
        print(len(self.start_states), len(self.train_states))
        self.dump_train_set()
        self.env.clear_skeletons()

    def training_iter(self):
        self.collect_dataset()
        self.train_inverse_dynamics()

    def evaluate(self, render, record_video=False, seed=None):
        if seed is None:
            seed = np.random.randint(100000)
        self.env.seed(seed)
        state = self.env.reset(record_video=record_video)
        total_error = 0
        max_error = 0
        total_score = 0
        DISCOUNT = 0.8
        n_steps = settings["n_steps"]
        targets = self.generate_targets(state, n_steps, settings["runway_length"])
        total_offset = 0
        num_successful_steps = 0
        for i in range(n_steps):
            target = targets[i]
            action = self.act(state, target)
            state = self.env.simulate(target, action, render=render)
            if type(state) == str:
                # The robot crashed
                print(state)
                break
            num_successful_steps += 1
            achieved_target = state[20:22]
            error = np.linalg.norm(achieved_target - target)
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

if __name__ == '__main__':
    from walker import TwoStepEnv
    env = TwoStepEnv(Simbicon)
    learn = LearnInverseDynamics(env)
    learn.load_train_set()
    #learn.training_iter()
    embed()
