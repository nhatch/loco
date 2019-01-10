import numpy as np
from IPython import embed
import pickle
import time
import gym

class RandomSearch:
    def __init__(self, runner, n_dirs, step_size=0.01, eps=0.05):
        self.runner = runner
        self.w_policy = self.runner.env.controller.action_scale() * 0.0

        self.n_dirs = n_dirs
        self.eps = eps
        self.step_size = step_size
        self.episodes = 0

    def sample_perturbation(self):
        delta = np.random.randn(np.prod(self.w_policy.shape))
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
        if self.eval(tol, render):
            return self.w_policy
        for i in range(max_iters):
            # TODO: should we "give up" if we don't see fast enough progress?
            grad = self.estimate_grad(self.w_policy)
            self.w_policy += self.step_size * grad
            self.step_size *= 0.8
            if self.eval(tol, render):
                return self.w_policy
        print("Max iters exceeded")
        return None

    def manual_search(self, action, next_target, next_heading, video_save_dir=None):
        action += self.w_policy
        self.runner.reset(video_save_dir)
        env = self.runner.env
        render = 0.7 if video_save_dir else 1.0
        env.simulate(self.runner.target, action=action, render=render)
        # Simulate the next step as well to get a sense of where that step left us.
        env.simulate(next_target, target_heading=next_heading, render=render)

    def eval(self, tol, render):
        ret = self.runner.run(self.w_policy, render=render)
        # The best possible score is 0
        return ret > -tol
