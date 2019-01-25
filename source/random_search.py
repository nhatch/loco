import numpy as np
from IPython import embed
import pickle
import time
import gym
from simbicon_params import PARAM_SCALE

class RandomSearch:
    def __init__(self, runner, settings):
        self.runner = runner
        self.w_policy = np.zeros(len(PARAM_SCALE))

        self.n_dirs = settings['n_dirs']
        self.tol = settings['tol']
        self.eps = settings.get('eps') or 0.1
        self.max_iters = settings.get('max_iters') or 5
        self.controllable_params = settings['controllable_params']

    def sample_perturbation(self):
        delta = np.random.randn(np.prod(self.w_policy.shape))
        delta *= self.controllable_params
        delta /= np.linalg.norm(delta)
        return delta.reshape(self.w_policy.shape)

    def estimate_grad(self, policy):
        grad = np.zeros_like(policy)
        rets = []
        for j in range(self.n_dirs):
            p = self.sample_perturbation()
            self.runner.reset()
            p_ret = self.runner.run(policy + self.eps*p)
            self.runner.reset()
            n_ret = self.runner.run(policy - self.eps*p)
            rets.append(p_ret)
            rets.append(n_ret)
            grad += (p_ret - n_ret) / self.n_dirs * p
        return grad / np.std(rets)

    def random_search(self, render=1.0, video_save_dir=None):
        if self.eval(render, video_save_dir):
            return self.w_policy
        for i in range(self.max_iters):
            # TODO: should we "give up" if we don't see fast enough progress?
            grad = self.estimate_grad(self.w_policy)
            self.w_policy += self.step_size * grad
            if self.eval(render, video_save_dir):
                return self.w_policy
        print("MAX ITERS EXCEEDED")
        return None

    def eval(self, render, video_save_dir):
        self.runner.reset(video_save_dir, render)
        ret = self.runner.run(self.w_policy)
        # The best possible score is 0
        print("Score: {:.4f}".format(ret))
        # This is a terrible hack. Might take some bizarre theory to justify it.
        # The idea is, the closer we are to the target, the smaller our next
        # step size should be.
        self.step_size = -ret
        return ret > -self.tol
