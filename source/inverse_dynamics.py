from IPython import embed
import numpy as np
import pickle
import os

from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from simbicon import Simbicon
from simbicon_params import *
from state import State
from step_learner import Runner
from random_search import RandomSearch

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
        self.train_features, self.train_responses = [], []
        self.n_action = len(self.env.controller.base_gait())
        self.n_dynamic = 22
        # TODO try some model more complicated than linear?
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

        # Evaluation settings
        self.dist_mean = 0.42
        self.dist_spread = 0.3
        self.n_steps = 16
        self.runway_length = 0.4

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
        starter = 0.3
        self.env.sdf_loader.put_grounds([[0,0,0]], runway_length=100)
        for i in range(n_resets):
            length = min_length + (max_length - min_length) * (i / n_resets)
            self.env.log("Starting trajectory {}".format(i))
            self.env.reset()
            # TODO should we include this first state? It will be very different from the rest.
            #start_states.append(self.env.robot_skeleton.x)
            target = np.array([0,0,0])
            for j in range(size):
                prev_target, target = target, np.array([starter + length * j, 0, 0])
                end_state, _ = self.env.simulate(target, render=0.1)
                # We need to collect the location of the previous target in order to
                # place stepping stones properly when resetting to this state.
                start_states.append(end_state)
        self.env.clear_skeletons()
        return start_states

    def dump_train_set(self):
        with open(TRAIN_FILENAME, 'wb') as f:
            pickle.dump((self.start_states, self.train_features, self.train_responses), f)

    def load_train_set(self):
        with open(TRAIN_FILENAME, 'rb') as f:
            self.start_states, self.train_features, self.train_responses = pickle.load(f)
        self.train_inverse_dynamics()

    def collect_dataset(self):
        self.env.clear_skeletons()
        indices = np.random.choice(range(len(self.start_states)), size=N_STATES_PER_ITER)
        for i,j in enumerate(indices):
            print("Exploring start state {} ({} / {})".format(j, i, len(indices)))
            self.collect_samples(self.start_states[j])
        print(len(self.start_states), len(self.train_features))
        self.dump_train_set()
        self.env.clear_skeletons()

    def collect_samples(self, start_state):
        target = self.generate_targets(start_state, 1)[0]
        mean_action, runner = self.learn_action(start_state, target)
        for _ in range(N_ACTIONS_PER_STATE):
            runner.reset()
            # TODO test whether these perturbations actually help
            perturbation = EXPLORATION_STD * np.random.randn(len(mean_action))
            perturbation *= self.env.controller.controllable_indices()
            action = mean_action + perturbation
            end_state, terminated = self.env.simulate(target, action)
            if not terminated:
                self.append_to_train_set(start_state, target, action, end_state)
                self.start_states.append(end_state)

    def learn_action(self, start_state, target):
        runner = Runner(self.env, start_state, target)
        rs = RandomSearch(runner, 4, step_size=0.1, eps=0.05)
        rs.w_policy = self.act(start_state, target) # Initialize with something reasonable
        rs.random_search(render=None)
        return rs.w_policy, runner

    def append_to_train_set(self, start_state, target, action, end_state):
        # The train set is a set of (features, action) pairs
        # where taking `action` when the environment features are `features`
        # will *exactly* hit the target with the swing foot.
        # Note that the target must be part of `features` and that
        # the `features` must capture all information necessary to reconstruct the
        # world state to verify that this action will indeed hit the target.
        # But it also must be reasonably low-dimensional because these are also the
        # features used in our ML algorithm. TODO: separate these two responsibilities.
        achieved_target = end_state.stance_heel_location()
        # TODO: This "hindsight retargeting" doesn't exactly work. The problem is that
        # moving the target platform to the new location actually changes end-of-step
        # detection (e.g. the toe hits something where before there was empty space).
        action[TX:TX+3] += target - achieved_target
        train_state = self.extract_features(start_state, achieved_target)
        self.train_features.append(train_state)
        self.train_responses.append(action)

    # TODO Is there a way to "mask out" some of these features rather than rewriting this code
    # every time? (e.g. adding/removing the Z dimension)
    def extract_features(self, state, target):
        centered_state = np.zeros(self.n_dynamic)

        centered_state[ 0:18] = state.pose()
        centered_state[18:20] = state.stance_heel_location()[:2]
        # We don't include state.swing_platform or state.stance_platform
        # because they seem less important
        # (and we are desperate to reduce problem dimension).
        centered_state[20:22] = target[:2]

        # Absolute location does not affect dynamics, so recenter all (x,y) coordinates
        # around the location of the starting platform.
        base = state.stance_platform()[:2]
        centered_state[ 0: 2] -= base # starting location of agent
        centered_state[18:20] -= base # starting stance heel location
        centered_state[20:22] -= base # ending platform location

        return centered_state

    def train_inverse_dynamics(self):
        X = np.array(self.train_features)
        y = np.array(self.train_responses)
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean) / self.X_scale_factor
        y = (y - self.y_mean) * self.y_scale_factor
        self.model.fit(X, y)

    def act(self, state, target):
        X = self.extract_features(state, target).reshape(1,-1)
        X = (X - self.X_mean) / self.X_scale_factor
        if hasattr(self.model, 'estimator_'):
            rescaled_action = self.model.predict(X).reshape(-1)
        else:
            rescaled_action = np.zeros(self.n_action)
        rescaled_action *= self.env.controller.controllable_indices()
        return rescaled_action / self.y_scale_factor + self.y_mean

    def generate_targets(self, start_state, num_steps, runway_length=None):
        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        targets = start_state.starting_platforms()
        next_target = targets[-1]
        for _ in range(num_steps):
            dx = self.dist_mean + self.dist_spread * (np.random.uniform() - 0.5)
            dy = np.random.uniform() * 0.0
            dz = 0.0
            next_target = next_target + [dx, dy, dz]
            targets.append(next_target)
        self.env.sdf_loader.put_grounds(targets, runway_length=runway_length)
        return targets[2:] # Don't include the starting platforms

    def training_iter(self):
        self.collect_dataset()
        self.train_inverse_dynamics()

    def evaluate(self, render=1.0, record_video=False, seed=None):
        if seed is None:
            seed = np.random.randint(100000)
        self.env.seed(seed)
        state = self.env.reset(record_video=record_video)
        total_error = 0
        max_error = 0
        total_score = 0
        DISCOUNT = 0.8
        targets = self.generate_targets(state, self.n_steps, self.runway_length)
        total_offset = 0
        num_successful_steps = 0
        for i in range(self.n_steps):
            target = targets[i]
            action = self.act(state, target)
            state, terminated = self.env.simulate(target, action, render=render)
            if terminated:
                break
            num_successful_steps += 1
            achieved_target = state.stance_heel_location()
            error = np.linalg.norm(achieved_target - target)
            if error > max_error:
                max_error = error
            total_error += error
            if error < 1:
                total_score += (1-error) * (DISCOUNT**i)
        if record_video:
            self.env.reset() # This ensures the video recorder is closed properly.
        result = {
                "total_score": total_score,
                "n_steps": num_successful_steps,
                "max_error": max_error,
                "total_error": total_error,
                }
        return result

def test_train_state_storage(env):
    learn = LearnInverseDynamics(env)
    start_state = np.random.choice(learn.start_states)
    learn.collect_samples(start_state)
    for i in range(len(learn.train_features)):
        features = learn.train_features[i]
        response = learn.train_responses[i]
        target = start_state.stance_platform().copy()
        target[0:2] += features[-2:]
        # TODO reconstruct start_state from features (in order to see things from the
        # perspective of the learning algorithm)
        runner = Runner(env, start_state, target)
        score = runner.run(response, render=1.0)
        print("Score:", score) # This score should be exactly zero.
        # But often it isn't; see todo in append_to_train_set

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    #test_train_state_storage(env)

    learn = LearnInverseDynamics(env)
    learn.load_train_set()
    learn.training_iter()
    embed()
