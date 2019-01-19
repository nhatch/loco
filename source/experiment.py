from IPython import embed
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from inverse_dynamics import LearnInverseDynamics

from test_inverse_dynamics import TRAIN_SETTINGS_3D, TRAIN_SETTINGS_3D_PRECISE, SETTINGS_3D_EASY, SETTINGS_3D_MEDIUM, SETTINGS_3D_HARD, SETTINGS_3D_HARDER

N_EVAL_TRAJECTORIES = 8


class Experiment:
    def __init__(self, env, name):
        self.settings = {
                "total_score": "blue",
                "max_error": "red",
                "n_steps": "black",
                }
        self.name = name
        self.learn = LearnInverseDynamics(env, self.name)
        self.iter = 0
        self.results = defaultdict(lambda: [])
        self.n_eval_trajectories = N_EVAL_TRAJECTORIES
        #self.run_evaluations()

    def checkpoint(self):
        # TODO add checkpointing (save model params, collected data, training curve statistics)
        pass

    def run_evaluations(self):
        results = defaultdict(lambda: [])
        for i in range(self.n_eval_trajectories):
            print("Starting evaluation", i)
            result = self.learn.evaluator.evaluate(self.learn.act, render=None)
            for k,v in result.items():
                results[k].append(v)
        for k,v in results.items():
            self.results[k].append(v)
        self.plot_results()
        self.learn.evaluate() # For human consumption

    def run_iters(self, n_iters, eval_settings, train_settings):
        self.learn.set_train_settings(train_settings)
        for i in range(n_iters):
            # TODO should we also run evaluations for easier settings?
            self.learn.evaluator.set_eval_settings(SETTINGS_3D_HARDER)
            self.run_evaluations()
            self.learn.evaluator.set_eval_settings(eval_settings)
            self.learn.training_iter()

    def plot_results(self):
        n_points = self.iter+1
        lines = []
        labels = []
        keys_to_plot = ['total_score', 'max_error']
        for k in keys_to_plot:
            data = np.array(self.results[k])
            x = range(data.shape[0])
            color = self.settings[k]
            line, = plt.plot(x, np.mean(data, 1), color=color)
            plt.fill_between(x, np.min(data, 1), np.max(data, 1), color=color, alpha=0.2)
            labels.append(k)
            lines.append(line)

        plt.title("Training curve")
        plt.xlabel("Number of data collection iterations")
        plt.ylabel("Foot placement error")
        plt.legend(lines, labels)

        sns.set_style('white')
        sns.despine()

        plt.savefig('{}.png'.format(self.name))
        plt.clf()

def load(env, name):
    learn = LearnInverseDynamics(env, name)
    learn.load_train_set()
    return learn

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    #env = SteppingStonesEnv()
    env = Simple3DEnv(Simbicon3D)
    ex = Experiment(env, "new_experiment")
    ex.run_iters(2, SETTINGS_3D_EASY, TRAIN_SETTINGS_3D)
    ex.run_iters(6, SETTINGS_3D_MEDIUM, TRAIN_SETTINGS_3D)
    ex.run_iters(6, SETTINGS_3D_MEDIUM_PRECISE, TRAIN_SETTINGS_3D_PRECISE)
    ex.run_iters(6, SETTINGS_3D_HARD, TRAIN_SETTINGS_3D)
    ex.run_iters(6, SETTINGS_3D_HARD_PRECISE, TRAIN_SETTINGS_3D_PRECISE)
    ex.run_iters(6, SETTINGS_3D_HARDER, TRAIN_SETTINGS_3D_PRECISE)
    ex.run_evaluations()
    #learn = load(env, 'my_experiment')
    embed()
