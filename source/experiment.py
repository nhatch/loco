from IPython import embed
import numpy as np
import pickle
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from inverse_dynamics import LearnInverseDynamics
import curriculum as cur

RESULTS_FMT = 'data/results_{}.pkl'

N_EVAL_TRAJECTORIES = 8
KEYS_TO_SAVE = ['total_reward', 'max_error', 'n_steps', 'seed']
KEYS_TO_PLOT = ['total_reward', 'max_error']

class Experiment:
    def __init__(self, env, name, final_eval_settings):
        self.settings = {
                "total_reward": "blue",
                "max_error": "red",
                "n_steps": "black",
                }
        self.name = name
        self.learn = LearnInverseDynamics(env, self.name)
        self.n_eval_trajectories = N_EVAL_TRAJECTORIES
        self.load(final_eval_settings)

    def save(self):
        fname = RESULTS_FMT.format(self.name)
        with open(fname, 'wb') as f:
            pickle.dump(self.results, f)

    def load(self, final_eval_settings):
        fname = RESULTS_FMT.format(self.name)
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
            self.plot_results(settings_name)
        self.save()
        #self.learn.evaluate() # For human consumption

    def run_iters(self, n_iters, eval_settings, train_settings):
        self.learn.set_train_settings(train_settings)
        for i in range(n_iters):
            self.learn.evaluator.set_eval_settings(eval_settings)
            self.learn.training_iter()
            self.run_evaluations()

    def plot_results(self, settings_name):
        lines = []
        labels = []
        for k in KEYS_TO_PLOT:
            data = np.array(self.results[settings_name][k])
            x = range(data.shape[0])
            color = self.settings[k]
            mean = np.mean(data, 1)
            line, = plt.plot(x, mean, color=color)
            #plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
            plt.fill_between(x, np.min(data, 1), np.max(data, 1), color=color, alpha=0.2)
            labels.append(k)
            lines.append(line)

        plt.title(settings_name)
        plt.xlabel("Number of data collection iterations")
        plt.ylabel("Foot placement error")
        plt.legend(lines, labels)

        sns.set_style('white')
        sns.despine()

        plt.savefig('{}_{}.png'.format(self.name, settings_name))
        plt.clf()

def ex_3D():
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    env = Simple3DEnv(Simbicon3D)
    ex = Experiment(env, "new_experiment", ['SETTINGS_3D_HARDER'])
    ex.run_evaluations()
    ex.run_iters(2, cur.SETTINGS_3D_EASY, cur.TRAIN_SETTINGS_3D)
    ex.run_iters(6, cur.SETTINGS_3D_MEDIUM, cur.TRAIN_SETTINGS_3D)
    ex.run_iters(6, cur.SETTINGS_3D_MEDIUM, cur.TRAIN_SETTINGS_3D_PRECISE)
    ex.run_iters(6, cur.SETTINGS_3D_HARD, cur.TRAIN_SETTINGS_3D)
    ex.run_iters(6, cur.SETTINGS_3D_HARD, cur.TRAIN_SETTINGS_3D_PRECISE)
    ex.run_iters(6, cur.SETTINGS_3D_HARDER, cur.TRAIN_SETTINGS_3D_PRECISE)
    embed()

def ex_2D():
    from stepping_stones_env import SteppingStonesEnv
    env = SteppingStonesEnv()
    ex = Experiment(env, "2D_quadratic_after_29", ['SETTINGS_2D_EASY', 'SETTINGS_2D_HARD'])
    #ex.run_iters(3, cur.SETTINGS_2D_EASY, cur.TRAIN_SETTINGS_2D)
    #ex.run_iters(10, cur.SETTINGS_2D_HARD, cur.TRAIN_SETTINGS_2D_PLUS)
    embed()

if __name__ == '__main__':
    ex_2D()
