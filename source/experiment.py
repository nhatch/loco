from IPython import embed
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from inverse_dynamics import LearnInverseDynamics

N_EVAL_TRAJECTORIES = 8


class Experiment:
    def __init__(self, env, name):
        self.settings = {
                "total_score": "blue",
                "max_error": "red",
                "n_steps": "black",
                }
        self.name = name
        self.learner = LearnInverseDynamics(env, self.name)
        self.iter = 0
        self.results = defaultdict(lambda: [])
        self.n_eval_trajectories = N_EVAL_TRAJECTORIES
        self.run_evaluations()

    def checkpoint(self):
        # TODO add checkpointing (save model params, collected data, training curve statistics)
        pass

    def run_evaluations(self):
        results = defaultdict(lambda: [])
        for i in range(self.n_eval_trajectories):
            print("Starting evaluation", i)
            result = self.learner.evaluate(render=None)
            for k,v in result.items():
                results[k].append(v)
        for k,v in results.items():
            self.results[k].append(v)
        self.plot_results()
        self.learner.evaluate() # For human consumption

    def run_iters(self, n_iters):
        for i in range(n_iters):
            self.iter += 1
            print("Starting training iteration", self.iter)
            self.learner.training_iter()
            self.run_evaluations()

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
    from simbicon import Simbicon
    env = SteppingStonesEnv()
    ex = Experiment(env, "my_experiment_perturbations")
    ex.run_iters(6)
    #learn = load(env, 'my_experiment')
    embed()
