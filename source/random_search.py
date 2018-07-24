import numpy as np
from IPython import embed
import pickle
import time
import gym

controllable_indices = [0, 1, 1, 1, 0, 0, 0, 0, 0,
                        1, 0, 0, 1, 0, 0, 0, 1, 1]

class RandomSearch:
    def __init__(self, env, runner, n_dirs, step_size=0.01, eps=0.05):
        self.runner = runner
        action_dim = env.action_space.shape[0]
        self.w_policy = np.zeros(action_dim)

        self.n_dirs = n_dirs
        self.eps = eps
        self.step_size = step_size
        self.episodes = 0

    def sample_perturbation(self):
        delta = np.random.randn(np.prod(self.w_policy.shape))
        delta *= controllable_indices
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

    def random_search(self, max_iters=10, tol=0.02, render=1.0):
        for i in range(max_iters):
            if self.eval(tol, render):
                return
            grad = self.estimate_grad(self.w_policy)
            self.w_policy += self.step_size * grad
            self.step_size *= 0.8
        print("Max iters exceeded")

    def eval(self, tol, render):
        ret = self.runner.run(self.w_policy, render=render)
        # The best possible score is 0
        return ret > -tol
