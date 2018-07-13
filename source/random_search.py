import numpy as np
from IPython import embed
import pickle
import time
import gym
from inverse_dynamics import controllable_indices

class RandomSearch:
    def __init__(self, env, runner, n_dirs, step_size=0.01, eps=0.05):
        self.runner = runner
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.w_policy = np.zeros((action_dim, obs_dim))

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
            seed = np.random.randint(1000)
            p = self.sample_perturbation()
            p_ret = self.runner.run(policy + self.eps*p, seed)
            n_ret = self.runner.run(policy - self.eps*p, seed)
            rets.append(p_ret)
            rets.append(n_ret)
            grad += (p_ret - n_ret) / self.n_dirs * p
        self.episodes += self.n_dirs*2
        return grad / np.std(rets)

    def random_search(self, steps = 1000):
        for i in range(steps):
            print(i)
            grad = self.estimate_grad(self.w_policy)
            self.w_policy += self.step_size * grad
            self.demo()
            self.step_size *= 0.8

    def save(self, name):
        self.runner.save(name)
        params = (self.w_policy, self.episodes, self.n_dirs, self.eps, self.step_size)
        f = open('saved_controllers/{}.pkl'.format(name), 'wb')
        pickle.dump(params, f)
        f.close()

    def load(self, name):
        self.runner.load(name)
        f = open('saved_controllers/{}.pkl'.format(name), 'rb')
        params = pickle.load(f)
        self.w_policy = params[0]
        self.episodes = params[1]
        self.n_dirs = params[2]
        self.eps = params[3]
        self.step_size = params[4]
        f.close()

    def demo(self):
        ret = self.runner.run(self.w_policy, seed=None, render=1)
        print(self.episodes, ret)

if __name__ == "__main__":
    # TODO this is broken now that we must provide an explicit "runner"
    print("Making env")
    env = gym.make('DartHopper-v1')
    rs = RandomSearch(env, 8, 0.01, 0.05)
    print("Starting search")
    rs.random_search(25)
    #rs.random_search(250)
    embed()
