from IPython import embed
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from inverse_dynamics import LearnInverseDynamics


class Experiment:
    def __init__(self, learner):
        self.learner = learner

    def repeat_eval(self, n_eval_repeats):
        avg_losses = []
        max_errors = []
        for k in range(n_eval_repeats):
            print("Starting evaluation", k)
            avg_loss, max_error = self.learner.evaluate()
            avg_losses.append(avg_loss)
            max_errors.append(max_error)
        return np.mean(avg_losses), np.std(avg_losses), np.mean(max_errors), np.std(max_errors)

    def multi_evaluate(self, n_iters=3, n_eval_repeats=8):
        results = np.zeros((n_iters+1, 4))
        results[0] = self.repeat_eval(n_eval_repeats)
        for i in range(n_iters):
            self.learner.training_iter()
            results[i+1] = self.repeat_eval(n_eval_repeats)
        self.plot_results(results)

    def plot_results(self, results):
        x = list(range(results.shape[0]))
        avg_loss_line, = plt.plot(x, results[:,0], color="red")
        max_error_line, = plt.plot(x, results[:,2], color="blue")
        plt.fill_between(x, results[:,0]-results[:,1], results[:,0]+results[:,1], color="red", alpha=0.2)
        plt.fill_between(x, results[:,2]-results[:,3], results[:,2]+results[:,3], color="blue", alpha=0.2)
        plt.savefig('{}.png'.format("thingy"))

if __name__ == '__main__':
    from walker import TwoStepEnv
    from simbicon import Simbicon
    env = TwoStepEnv(Simbicon)
    learn = LearnInverseDynamics(env)
    ex = Experiment(learn)
    ex.multi_evaluate()
    embed()
