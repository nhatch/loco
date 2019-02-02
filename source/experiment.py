from IPython import embed
import sys
import numpy as np
import pickle
import os.path
import os
from collections import defaultdict
from inverse_dynamics import LearnInverseDynamics
from rs_baseline import RandomSearchBaseline
import curriculum as cur

DIR_FMT = 'data/{}/'

N_EVAL_TRAJECTORIES = 8
KEYS_TO_SAVE = ['total_reward', 'max_error', 'n_steps', 'seed']

class Experiment:
    def __init__(self, env, learner_class, name, final_eval_settings):
        self.name = name
        self.learn = learner_class(env, self.name)
        self.n_eval_trajectories = N_EVAL_TRAJECTORIES
        self.load(final_eval_settings)

    def save_filename(self, filename):
        dirname = DIR_FMT.format(self.name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        return dirname + filename

    def save(self):
        fname = self.save_filename('results.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, final_eval_settings):
        fname = self.save_filename('results.pkl')
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.results = pickle.load(f)
            if len(self.results[final_eval_settings[0]][KEYS_TO_SAVE[0]]) > 1:
                self.learn.load_train_set()
        else:
            self.results = {}
            for settings_name in final_eval_settings:
                d = {}
                for k in KEYS_TO_SAVE:
                    d[k] = []
                self.results[settings_name] = d
            self.run_evaluations()

    def revert_to_iteration(self, iteration, new_name):
        self.name = new_name
        for settings_name in self.results.keys():
            for k in KEYS_TO_SAVE:
                self.results[settings_name][k] = self.results[settings_name][k][:iteration+1]
        self.save()
        self.learn.revert_to_iteration(iteration, new_name)

    def run_evaluations(self):
        for settings_name in self.results.keys():
            settings = cur.__dict__[settings_name]
            self.learn.evaluator.set_eval_settings(settings)
            results = defaultdict(lambda: [])
            for i in range(self.n_eval_trajectories):
                print("Starting evaluation", i)
                result = self.learn.evaluate(render=None)
                for k in KEYS_TO_SAVE:
                    results[k].append(result[k])
            for k,v in results.items():
                self.results[settings_name][k].append(v)
        self.save()
        #self.learn.evaluate() # For human consumption

    def run_iters(self, n_iters, eval_settings, train_settings):
        self.learn.set_train_settings(train_settings)
        for i in range(n_iters):
            self.learn.evaluator.set_eval_settings(eval_settings)
            self.learn.training_iter()
            self.run_evaluations()

def ex_3D(uq_id):
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    env = Simple3DEnv(Simbicon3D)
    ex = Experiment(env, LearnInverseDynamics, "3D_test_"+uq_id, ['SETTINGS_3D_MEDIUM', 'SETTINGS_3D_HARD'])
    ex.run_iters(5, cur.SETTINGS_3D_EASY, cur.TRAIN_SETTINGS_3D)
    ex.run_iters(15, cur.SETTINGS_3D_MEDIUM, cur.TRAIN_SETTINGS_3D)
    ex.run_iters(6, cur.SETTINGS_3D_HARD, cur.TRAIN_SETTINGS_3D)
    ex.run_iters(12, cur.SETTINGS_3D_HARD, cur.TRAIN_SETTINGS_3D_PLUS)

def ex_2D_cim(uq_id):
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    ex = Experiment(env, LearnInverseDynamics, "cim_final_"+uq_id, ['SETTINGS_2D_HARD'])
    ex.run_iters(3, cur.SETTINGS_2D_EASY, cur.TRAIN_SETTINGS_2D)
    ex.run_iters(18, cur.SETTINGS_2D_HARD, cur.TRAIN_SETTINGS_2D)
    ex.run_iters(12, cur.SETTINGS_2D_HARD, cur.TRAIN_SETTINGS_2D_PLUS)

def ex_2D_cim_easy(uq_id):
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    ex = Experiment(env, LearnInverseDynamics, "cim_final_easy_"+uq_id, ['SETTINGS_2D_EASY'])
    ex.run_iters(5, cur.SETTINGS_2D_EASY, cur.TRAIN_SETTINGS_2D)

def ex_2D_nocur(uq_id):
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    ex = Experiment(env, LearnInverseDynamics, "nocur_final_"+uq_id, ['SETTINGS_2D_HARD'])
    ex.run_iters(1, cur.SETTINGS_2D_HARD, cur.TRAIN_SETTINGS_2D_NOCUR_FIRST)
    ex.run_iters(8, cur.SETTINGS_2D_HARD, cur.TRAIN_SETTINGS_2D_NOCUR_NEXT)

def ex_2D_rs(uq_id):
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    ex = Experiment(env, RandomSearchBaseline, "rs_final_"+uq_id, ['SETTINGS_2D_EASY'])
    ex.run_iters(10, cur.SETTINGS_2D_EASY, cur.TRAIN_SETTINGS_BASELINE_NOEXPERT)

if __name__ == '__main__':
    UQ_ID = sys.argv[1]
    ex_3D(UQ_ID)
