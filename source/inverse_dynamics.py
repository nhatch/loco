from IPython import embed
import numpy as np
import pickle
import os
import time

from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import simbicon_params as sp
from state import reconstruct_state
from step_learner import Runner
from random_search import RandomSearch
from evaluator import Evaluator

TRAIN_FMT = 'data/train_{}.pkl'

RIDGE_ALPHA = 0.1

class LearnInverseDynamics:
    def __init__(self, env, name=''):
        self.env = env
        self.evaluator = Evaluator(env)
        self.name = name
        self.train_features, self.train_responses = [], []
        self.history = []
        # TODO try some model more complicated than linear?
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        #model = LinearRegression(fit_intercept=False)
        #model = make_pipeline(PolynomialFeatures(2), model) # quadratic
        #model = RANSACRegressor(base_estimator=model, residual_threshold=2.0)
        self.model = model
        self.is_fitted = False # For some dang reason sklearn doesn't track this itself
        self.total_steps = 0
        self.total_failed_annotations = 0
        self.total_train_time = 0.0

    def set_train_settings(self, settings):
        self.train_settings = settings

    def dump_train_set(self):
        fname = TRAIN_FMT.format(self.name)
        with open(fname, 'wb') as f:
            pickle.dump((self.history, self.train_features, self.train_responses), f)

    def load_train_set(self):
        fname = TRAIN_FMT.format(self.name)
        with open(fname, 'rb') as f:
            self.history, self.train_features, self.train_responses = pickle.load(f)
        self.revert_to_iteration(len(self.history), self.name)

    def revert_to_iteration(self, iteration, new_name):
        # Resets so the next iteration index will be `iteration`
        self.history = self.history[:iteration]
        h = self.history[-1]
        index = h[0]
        self.train_features = self.train_features[:index]
        self.train_responses = self.train_responses[:index]
        self.total_steps = h[1]
        self.total_failed_annotations = h[2]
        self.total_train_time = h[3]
        if new_name is not None:
            self.name = new_name
        self.dump_train_set()
        self.evaluator.set_eval_settings(h[4])
        self.set_train_settings(h[5])
        self.train_inverse_dynamics()

    def training_iter(self):
        print("STARTING TRAINING ITERATION", len(self.history))
        t = time.time()

        experience = self.run_trajectories()
        self.expert_annotate(experience)
        self.train_inverse_dynamics()

        self.total_train_time += time.time() - t
        self.history.append((
            len(self.train_features),
            self.total_steps,
            self.total_failed_annotations,
            self.total_train_time,
            self.evaluator.eval_settings,
            self.train_settings))
        print("Size of train set:", len(self.train_features))
        self.dump_train_set()

    def run_trajectories(self):
        experience = []
        s = self.train_settings
        for i in range(s['n_trajectories']):
            r = self.evaluate(max_intolerable_steps=s['max_intolerable_steps'])
            experience += self.evaluator.experience
            self.total_steps += r['n_steps']
        return experience

    def expert_annotate(self, experience):
        self.env.clear_skeletons()
        total = len(experience)
        for i, (state, _, reward) in enumerate(experience):
            print("Finding expert label for state {}/{}".format(i, total))
            if reward < 1-self.train_settings['tol']:
                self.label_state(state)
        self.env.clear_skeletons()

    def label_state(self, features):
        mean_action = self.learn_action(features)
        if mean_action is None:
            # Random search couldn't find a good enough action; don't use this for training.
            self.total_failed_annotations += 1
            return
        # The train set is a set of (features, action) pairs
        # where taking `action` when the environment features are `features`
        # will hit the target with the swing foot (within 5 cm).
        self.train_features.append(features)
        # We don't need to mirror the action, because Simbicon3D already handled that
        # during random search.
        self.train_responses.append(mean_action)

    def learn_action(self, features):
        start_state, target = reconstruct_state(features, self.env.consts())
        runner = Runner(self.env, start_state, target,
                use_stepping_stones=self.evaluator.eval_settings['use_stepping_stones'])
        rs = RandomSearch(runner, self.train_settings)
        runner.reset()
        rs.w_policy = self.act(features) # Initialize with something reasonable
        # TODO put max_iters and tol in the object initialization params instead
        w_policy = rs.random_search(render=1)
        self.total_steps += runner.n_runs
        return w_policy

    def train_inverse_dynamics(self):
        c = self.env.consts()
        # TODO investigate removing more features to reduce problem dimension
        s = self.train_settings
        X = np.array(self.train_features) * s['observable_features']
        y = np.array(self.train_responses) * s['controllable_params']
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean)
        y = (y - self.y_mean)
        self.model.fit(X, y)
        self.is_fitted = True

    def act(self, features):
        action = np.zeros(sp.N_PARAMS)
        if self.is_fitted:
            s = self.train_settings
            c = self.env.consts()
            X = features.reshape(1,-1)
            X = (X - self.X_mean)
            prediction = self.model.predict(X).reshape(-1)
            action = prediction + self.y_mean
            # Simbicon3D will handle mirroring the action if necessary
        return action

    def evaluate(self, **kwargs):
        return self.evaluator.evaluate(self.act, **kwargs)
