import numpy as np
from IPython import embed
import pickle
import time
import gym

class RandomSearch:
    def __init__(self, runner, n_dirs, step_size=0.01, eps=0.05):
        self.runner = runner
        self.w_policy = self.runner.env.controller.controllable_indices() * 0.0

        self.n_dirs = n_dirs
        self.eps = eps
        self.step_size = step_size
        self.episodes = 0

    def sample_perturbation(self):
        delta = np.random.randn(np.prod(self.w_policy.shape))
        delta *= self.runner.env.controller.controllable_indices()
        delta /= np.linalg.norm(delta)
        return delta.reshape(self.w_policy.shape)

    def estimate_grad(self, policy):
        grad = np.zeros_like(policy)
        rets = []
        for j in range(self.n_dirs):
            p = self.sample_perturbation()
            p_ret = self.runner.run(policy + self.eps*p)
            n_ret = self.runner.run(policy - self.eps*p)
            rets.append(p_ret)
            rets.append(n_ret)
            grad += (p_ret - n_ret) / self.n_dirs * p
        self.episodes += self.n_dirs*2
        return grad / np.std(rets)

    def random_search(self, max_iters=10, tol=0.05, render=1.0):
        for i in range(max_iters):
            # TODO: should we "give up" if we don't see fast enough progress?
            if self.eval(tol, render):
                return self.w_policy
            grad = self.estimate_grad(self.w_policy)
            self.w_policy += self.step_size * grad
            self.step_size *= 0.8
        print("Max iters exceeded")
        return None

    def eval(self, tol, render):
        ret = self.runner.run(self.w_policy, render=render, record_video=self.record_video)
        # The best possible score is 0
        return ret > -tol
