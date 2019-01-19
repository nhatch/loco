from IPython import embed
import numpy as np
import pickle
import os

from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from simbicon_params import *
from state import State
from step_learner import Runner
from random_search import RandomSearch

TRAIN_FMT = 'data/train_{}.pkl'

RIDGE_ALPHA = 0.1
N_TRAJECTORIES = 3
EARLY_TERMINATION = 3

class LearnInverseDynamics:
    def __init__(self, env, exp_name=''):
        self.env = env
        self.exp_name = exp_name
        self.train_features, self.train_responses = [], []
        self.history = []
        self.n_action = len(PARAM_SCALE)
        self.n_dynamic = len(self.env.consts().observable_features)
        # TODO try some model more complicated than linear?
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        #model = LinearRegression(fit_intercept=False)
        #model = make_pipeline(PolynomialFeatures(2), model) # quadratic
        #model = RANSACRegressor(base_estimator=model, residual_threshold=2.0)
        self.model = model
        self.is_fitted = False # For some dang reason sklearn doesn't track this itself

    def set_eval_settings(self, settings):
        self.eval_settings = settings
        if self.eval_settings['use_stepping_stones']:
            self.env.sdf_loader.ground_width = self.eval_settings['ground_width']
            self.env.sdf_loader.ground_length = self.eval_settings['ground_length']
        else:
            self.env.sdf_loader.ground_width = 2.0
            self.env.sdf_loader.ground_length = 10.0
        self.env.clear_skeletons()

    def set_train_settings(self, settings):
        if settings.get('controllable_params') is None:
            settings['controllable_params'] = self.env.controller.default_controllable_params()
        self.env.controller.set_controllable_params(settings['controllable_params'])
        self.train_settings = settings

    def dump_train_set(self):
        fname = TRAIN_FMT.format(self.exp_name)
        with open(fname, 'wb') as f:
            pickle.dump((self.history, self.train_features, self.train_responses), f)

    def load_train_set(self):
        fname = TRAIN_FMT.format(self.exp_name)
        with open(fname, 'rb') as f:
            self.history, self.train_features, self.train_responses = pickle.load(f)
        self.train_inverse_dynamics()

    def revert_to_iteration(self, iteration, new_exp_name=None):
        # Resets so the next iteration index will be `iteration`
        self.history = self.history[:iteration]
        index = self.history[iteration-1][0]
        self.train_features = self.train_features[:index]
        self.train_responses = self.train_responses[:index]
        if new_exp_name is not None:
            self.exp_name = new_exp_name
        self.dump_train_set()
        self.train_inverse_dynamics()

    def training_iter(self):
        print("STARTING TRAINING ITERATION", len(self.history))
        states_to_label = self.run_trajectories()
        self.label_states(states_to_label)
        self.train_inverse_dynamics()

    def run_trajectories(self):
        states_to_label = []
        for i in range(N_TRAJECTORIES):
            r = self.evaluate(early_termination=EARLY_TERMINATION)
            states_to_label += r['failed_steps']
        return states_to_label

    def label_states(self, states_to_label):
        self.env.clear_skeletons()
        total = len(states_to_label)
        for i, (state, target) in enumerate(states_to_label):
            print("Finding expert label for state {}/{}".format(i, total))
            self.label_state(state, target)
        self.history.append((len(self.train_features), self.eval_settings, self.train_settings))
        print("Size of train set:", len(self.train_features))
        self.dump_train_set()
        self.env.clear_skeletons()

    def label_state(self, start_state, target):
        mean_action = self.learn_action(start_state, target)
        if mean_action is None:
            # Random search couldn't find a good enough action; don't use this for training.
            return
        self.append_to_train_set(start_state, target, mean_action)

    def learn_action(self, start_state, target):
        runner = Runner(self.env, start_state, target, use_stepping_stones=self.eval_settings['use_stepping_stones'])
        rs = RandomSearch(runner, self.train_settings)
        runner.reset()
        rs.w_policy = self.act(start_state, target) # Initialize with something reasonable
        # TODO put max_iters and tol in the object initialization params instead
        w_policy = rs.random_search(render=1)
        return w_policy

    def append_to_train_set(self, start_state, target, action):
        # The train set is a set of (features, action) pairs
        # where taking `action` when the environment features are `features`
        # will hit the target with the swing foot (within 5 cm).
        train_state = start_state.extract_features(target)
        self.train_features.append(train_state)
        # We don't need to mirror the action, because Simbicon3D already handled that
        # during random search.
        self.train_responses.append(action)

    def train_inverse_dynamics(self):
        c = self.env.consts()
        # TODO investigate removing more features to reduce problem dimension
        X = np.array(self.train_features)[:,c.observable_features]
        y = np.array(self.train_responses)[:,self.env.controller.controllable_params]
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean)
        y = (y - self.y_mean)
        self.model.fit(X, y)
        self.is_fitted = True

    def act(self, state, target):
        action = np.zeros(self.n_action)
        if self.is_fitted:
            c = self.env.consts()
            X = state.extract_features(target).reshape(1,-1)[:,c.observable_features]
            X = (X - self.X_mean)
            prediction = self.model.predict(X).reshape(-1)
            prediction = prediction + self.y_mean
            action[self.env.controller.controllable_params] = prediction
            # Simbicon3D will handle mirroring the action if necessary
        return action

    def generate_targets(self, start_state, num_steps, dist_mean=None, dist_spread=None):
        s = self.eval_settings
        if dist_mean is None:
            dist_mean = s['dist_mean']
        if dist_spread is None:
            dist_spread = s['dist_spread']
        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        targets = start_state.starting_platforms()
        next_target = targets[-1]
        # TODO should we make the first step a bit shorter, since we're starting from "rest"?
        for i in range(num_steps):
            dx = dist_mean + dist_spread * (np.random.uniform() - 0.5)
            dy = s['y_mean'] + s['y_spread'] * (np.random.uniform() - 0.5)
            dz = s['z_mean'] + s['z_spread'] * (np.random.uniform() - 0.5)
            if i % 2 == 1:
                dz *= -1
            next_target = next_target + [dx, dy, dz]
            targets.append(next_target)
        # Return value includes the two starting platforms
        return targets

    def evaluate(self, render=1.0, video_save_dir=None, seed=None, early_termination=None):
        s = self.eval_settings
        state = self.env.reset(video_save_dir=video_save_dir, seed=seed, random=0.005, render=render)
        total_error = 0
        max_error = 0
        total_score = 0
        DISCOUNT = 0.8
        targets = self.generate_targets(state, s['n_steps'])
        if s['use_stepping_stones']:
            self.env.sdf_loader.put_grounds(targets)
        else:
            self.env.sdf_loader.put_grounds(targets[:1])
        total_offset = 0
        num_successful_steps = 0
        failed_steps = []
        for i in range(s['n_steps']):
            target = targets[2+i]
            action = self.act(state, target)
            end_state, terminated = self.env.simulate(target, target_heading=0.0, action=action)
            error = np.linalg.norm(end_state.stance_heel_location() - target)
            if error > self.train_settings['tol']:
                failed_steps.append((state, target))
                terminate_early = (early_termination and len(failed_steps) >= early_termination)
                terminated = terminated or terminate_early
            if error > max_error:
                max_error = error
            total_error += error
            if error < 1:
                total_score += (1-error) * (DISCOUNT**i)
            if terminated:
                break
            state = end_state
            num_successful_steps += 1
        if video_save_dir:
            self.env.close_video_recorder()
        result = {
                "total_score": total_score,
                "n_steps": num_successful_steps,
                "max_error": max_error,
                "total_error": total_error,
                "failed_steps": failed_steps,
                }
        return result
