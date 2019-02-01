from IPython import embed
import numpy as np
import pickle
import os
import time

import simbicon_params as sp
from state import reconstruct_state
from step_learner import Runner
from random_search import RandomSearch
from evaluator import Evaluator

TRAIN_FMT = 'data/{}/history.pkl'
import curriculum as cur

def coef_to_fn(coef):
    def act(features):
        X = np.concatenate((features, [1.0]))
        return np.dot(coef, X)
    return act

class EpisodeRunner:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.n_steps = 0
        self.seed = np.random.randint(100000)

    def run(self, coef):
        policy = coef_to_fn(coef)
        # TODO use shaped total reward
        r = self.evaluator.evaluate(policy,
                seed=self.seed, render=self.render, video_save_dir=self.video_save_dir)
        ret = r['total_reward'] / 16 - 1 # Max ret of 0, min of -1
        self.n_steps += r['n_steps']
        return ret

    def reset(self, video_save_dir=None, render=None):
        self.video_save_dir = video_save_dir
        self.render = render

class RandomSearchBaseline:
    def __init__(self, env, name=''):
        self.env = env
        self.evaluator = Evaluator(env)
        self.name = name
        self.history = []
        self.total_steps = 0
        self.total_train_time = 0.0
        self.coef_ = np.zeros_like(cur.TRAIN_SETTINGS_BASELINE_NOEXPERT['controllable_params'], dtype=np.float32)

    def set_train_settings(self, settings):
        self.train_settings = settings
        # TODO implement switching to a quadratic model....
        # ...I have no idea how one would do this

    def dump_train_set(self):
        fname = TRAIN_FMT.format(self.name)
        with open(fname, 'wb') as f:
            pickle.dump(self.history, f)

    def load_train_set(self):
        fname = TRAIN_FMT.format(self.name)
        with open(fname, 'rb') as f:
            self.history = pickle.load(f)
        self.revert_to_iteration(len(self.history), self.name)

    def revert_to_iteration(self, iteration, new_name):
        # Resets so the next iteration index will be `iteration`
        self.history = self.history[:iteration]
        h = self.history[-1]
        self.coef_ = h[0]
        self.total_steps = h[1]
        self.total_train_time = h[2]
        if new_name is not None:
            self.name = new_name
        self.dump_train_set()
        self.evaluator.set_eval_settings(h[3])
        self.set_train_settings(h[4])

    def training_iter(self):
        print("STARTING TRAINING ITERATION", len(self.history))
        t = time.time()
        self.random_search()
        self.random_search() # Do it twice so evaluations are only 1/3 of compute time
        self.total_train_time += time.time() - t

        self.history.append((
            self.coef_,
            self.total_steps,
            self.total_train_time,
            self.evaluator.eval_settings,
            self.train_settings.copy()))
        print("Total steps:", self.total_steps)
        self.dump_train_set()

    def random_search(self):
        runner = EpisodeRunner(self.evaluator)
        rs = RandomSearch(runner, self.train_settings)
        rs.step_size = self.train_settings['rs_eps']
        grad = rs.estimate_grad(self.coef_)
        self.coef_ += self.train_settings['rs_step_size'] * grad
        self.total_steps += runner.n_steps
        self.train_settings['rs_eps'] *= 0.9
        self.train_settings['rs_step_size'] *= 0.9

    def evaluate(self, **kwargs):
        return self.evaluator.evaluate(coef_to_fn(self.coef_), **kwargs)
