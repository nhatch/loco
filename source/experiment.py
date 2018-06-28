from IPython import embed
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_dynamics import LearnInverseDynamics

N_EVAL_TRAJECTORIES = 8


class Experiment:
    def __init__(self, name, learner):
        self.learner = learner
        self.name = name
        self.iter = 0
        self.avg_losses = []
        self.max_errors = []
        self.n_eval_trajectories = N_EVAL_TRAJECTORIES
        self.run_evaluations()

    def checkpoint(self):
        # TODO add checkpointing (save model params, collected data, training curve statistics)
        pass

    def run_evaluations(self):
        avg_losses = []
        max_errors = []
        for k in range(self.n_eval_trajectories):
            print("Starting evaluation", k)
            avg_loss, max_error = self.learner.evaluate()
            avg_losses.append(avg_loss)
            max_errors.append(max_error)
        self.avg_losses.append(avg_losses)
        self.max_errors.append(max_errors)

    def run_iters(self, n_iters):
        for i in range(n_iters):
            self.learner.training_iter()
            self.run_evaluations()
            self.iter += 1
        self.plot_results()

    def plot_results(self):
        n_points = self.iter+1
        avg_losses = np.array(self.avg_losses)
        max_errors = np.array(self.max_errors)
        x = range(n_points)
        avg_loss_line, = plt.plot(x, np.mean(avg_losses, 1), color="red")
        max_error_line, = plt.plot(x, np.mean(max_errors, 1), color="blue")
        plt.fill_between(x, np.min(avg_losses, 1), np.max(avg_losses, 1), color="red", alpha=0.2)
        plt.fill_between(x, np.min(max_errors, 1), np.max(max_errors, 1), color="blue", alpha=0.2)

        plt.title("Training curve")
        plt.xlabel("Number of data collection iterations")
        plt.ylabel("Average loss / maximum placement error")
        plt.legend([avg_loss_line, max_error_line], ["Average loss", "Max error"])

        sns.set_style('white')
        sns.despine()

        plt.savefig('{}.png'.format(self.name))
        plt.clf()

if __name__ == '__main__':
    from walker import TwoStepEnv
    from simbicon import Simbicon
    env = TwoStepEnv(Simbicon)
    learn = LearnInverseDynamics(env)
    ex = Experiment("my_experiment", learn)
    ex.run_iters(0)
    embed()
