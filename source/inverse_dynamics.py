from IPython import embed
import numpy as np
import pickle
import os

from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from simbicon import Simbicon
from simbicon_params import *
from state import State
from step_learner import Runner
from random_search import RandomSearch

TRAIN_FMT = 'data/train_{}.pkl'

RIDGE_ALPHA = 0.1

class LearnInverseDynamics:
    def __init__(self, env, exp_name=''):
        self.env = env
        self.exp_name = exp_name
        self.train_features, self.train_responses = [], []
        self.history = []
        self.n_action = len(self.env.controller.action_params())
        self.n_dynamic = sum(self.env.consts().observable_features)
        # TODO try some model more complicated than linear?
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        #model = LinearRegression(fit_intercept=False)
        #model = make_pipeline(PolynomialFeatures(2), model) # quadratic
        #model = RANSACRegressor(base_estimator=model, residual_threshold=2.0)
        self.model = model
        self.is_fitted = False # For some dang reason sklearn doesn't track this itself
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

    def set_eval_settings(self, settings):
        self.eval_settings = settings

    def dump_train_set(self):
        fname = TRAIN_FMT.format(self.exp_name)
        with open(fname, 'wb') as f:
            pickle.dump((self.history, self.train_features, self.train_responses), f)

    def load_train_set(self):
        fname = TRAIN_FMT.format(self.exp_name)
        with open(fname, 'rb') as f:
            self.history, self.train_features, self.train_responses = pickle.load(f)
        self.train_inverse_dynamics()

    def training_iter(self):
        if self.eval_settings.get('ground_width'):
            self.env.sdf_loader.ground_width = self.eval_settings['ground_width']
        if self.eval_settings.get('ground_length'):
            self.env.sdf_loader.ground_length = self.eval_settings['ground_length']
        states_to_label = self.run_trajectories()
        self.label_states(states_to_label)
        self.train_inverse_dynamics()

    def run_trajectories(self):
        states_to_label = []
        for i in range(1):
            r = self.evaluate()
            states_to_label += r['failed_steps']
        return states_to_label

    def label_states(self, states_to_label):
        self.env.clear_skeletons()
        total = len(states_to_label)
        for i, (state, target) in enumerate(states_to_label):
            print("Finding expert label for state {}/{}".format(i, total))
            self.label_state(state, target)
        self.history.append((len(self.train_features), self.eval_settings))
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
        runner = Runner(self.env, start_state, target)
        if self.env.is_3D:
            rs = RandomSearch(runner, 8, step_size=0.2, eps=0.1)
        else:
            rs = RandomSearch(runner, 4, step_size=0.1, eps=0.1)
        runner.reset()
        rs.w_policy = self.act(target) # Initialize with something reasonable
        # TODO put max_iters and tol in the object initialization params instead
        w_policy = rs.random_search(max_iters=5, tol=0.05, render=1)
        return w_policy

    def append_to_train_set(self, start_state, target, action):
        # The train set is a set of (features, action) pairs
        # where taking `action` when the environment features are `features`
        # will hit the target with the swing foot (within 5 cm).
        train_state = self.extract_features(start_state, target)
        self.train_features.append(train_state)
        self.train_responses.append(action)

    def extract_features(self, state, target):
        # Combines state and target into a single vector, and discards some information
        # that does not affect the dynamics (such as absolute location).
        # That this vector is still very high dimenstional for debugging purposes.
        # When actually training the model, many of these features are discarded.
        c = self.env.consts()
        centered_state = np.zeros(2*c.Q_DIM + 3*3)

        # Recenter all (x,y,z) coordinates around the location of the starting platform.
        # TODO: also rotate so absolute yaw is 0.
        base = state.stance_platform()
        centered_state[ 0:-9] = state.pose()
        centered_state[ 0: 3] -= base # starting location of agent
        centered_state[-9:-6] = state.stance_heel_location() - base
        centered_state[-6:-3] = state.swing_platform() - base
        centered_state[-3:  ] = target - base
        return centered_state

    def reconstruct_state(self, features):
        # At the moment, used only for debugging.
        raw_state = features.copy()
        raw_state[-3:  ] = features[-6:-3]
        raw_state[-6:-3] = [0,0,0]
        target = features[-3:].copy()
        if self.env.is_3D:
            c = self.env.consts()
            raw_state[[c.Y, -2, -5, -8]] -= 0.9
            target[c.Y] -= 0.9
        return State(raw_state), target

    def train_inverse_dynamics(self):
        c = self.env.consts()
        # TODO investigate removing more features to reduce problem dimension
        X = np.array(self.train_features)[:,c.observable_features]
        y = np.array(self.train_responses)
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean) / self.X_scale_factor
        y = (y - self.y_mean) * self.y_scale_factor
        self.model.fit(X, y)
        self.is_fitted = True

    def act(self, target):
        c = self.env.consts()
        state = self.env.current_observation()
        if self.env.controller.swing_idx == c.LEFT_IDX:
            # `state` has been mirrored, but `target` has not.
            target = target * np.array([1,1,-1])
        X = self.extract_features(state, target).reshape(1,-1)[:,c.observable_features]
        X = (X - self.X_mean) / self.X_scale_factor
        if self.is_fitted:
            action = self.model.predict(X).reshape(-1)
        else:
            action = np.zeros(self.n_action)
        # Simbicon3D will handle mirroring the action if necessary
        return action / self.y_scale_factor + self.y_mean

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

    def evaluate(self, render=1.0, video_save_dir=None, seed=None):
        s = self.eval_settings
        state = self.env.reset(video_save_dir=video_save_dir, seed=seed, random=0.005, render=render)
        total_error = 0
        max_error = 0
        total_score = 0
        DISCOUNT = 0.8
        targets = self.generate_targets(state, s['n_steps'])
        if s['use_stepping_stones']:
            self.env.sdf_loader.put_grounds(targets, runway_length=s['runway_length'])
        else:
            self.env.sdf_loader.put_grounds(targets[:1], runway_length=s['runway_length'])
        total_offset = 0
        num_successful_steps = 0
        failed_steps = []
        for i in range(s['n_steps']):
            target = targets[2+i]
            action = self.act(target)
            start_state = state
            state, terminated = self.env.simulate(target, target_heading=0.0, action=action)
            error = np.linalg.norm(state.stance_heel_location() - state.stance_platform())
            if error > 0.02:
                failed_steps.append((start_state, state.stance_platform()*[1,1,-1]))
            if error > max_error:
                max_error = error
            total_error += error
            if error < 1:
                total_score += (1-error) * (DISCOUNT**i)
            if terminated or len(failed_steps) > 2:
                break
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
