import numpy as np
from IPython import embed
import pickle
import time
import gym

SHIFT = 0.0

class Whitener:
    def __init__(self, env, record_video=False):
        self.N = 0
        self.env = env
        obs_dim = env.observation_space.shape[0]
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)

        if record_video:
            n = int(time.time())
            m = gym.wrappers.Monitor(env, 'monitoring/{}'.format(n),
                    video_callable=lambda _: True)
            self.env = m

    def close(self):
        if type(self.env) == gym.wrappers.Monitor:
            self.env.close()

    def update_stats(self, obs):
        mean = (self.N * self.mean + obs) / (self.N+1)
        var = (self.N * self.var + self.N * (self.mean - mean)**2 + (obs - mean)**2) / (self.N+1)
        self.N = self.N+1
        self.mean = mean
        self.var = var

    def whiten_observation(self, obs, stats):
        if stats:
            self.update_stats(obs)
        var_to_use = self.var + (self.var == 0) # Avoid dividing by zero
        return (obs - self.mean) / var_to_use**(0.5)

    def run_trajectory(self, controller, seed, visual=True, stats=True, shift=SHIFT):
        self.env.seed(seed)
        observation = self.env.reset()
        obs = self.whiten_observation(observation, stats)
        rewards = []
        while True:
            observation, reward, done, info = self.env.step(controller(obs))
            obs = self.whiten_observation(observation, stats)
            if visual:
                self.env.render()
            rewards.append(reward - shift) # Subtract the "alive_bonus"
            if done:
                return sum(rewards)

    def rt(self, controller_matrix, seed=None, visual=False, stats=True, shift=SHIFT):
        c = lambda obs: np.dot(controller_matrix, obs)
        return self.run_trajectory(c, seed, visual, stats, shift)

class RandomSearch:
    def __init__(self, env, n_dirs, step_size=0.01, eps=0.05):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.w_policy = np.zeros((action_dim, obs_dim))

        self.whitener = Whitener(env)
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
            seed = np.random.randint(1000)
            p = self.sample_perturbation()
            p_ret = self.whitener.rt(policy + self.eps*p, seed)
            n_ret = self.whitener.rt(policy - self.eps*p, seed)
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

    def save(self, name):
        w = self.whitener
        params = (self.w_policy, self.episodes, self.n_dirs, self.eps, self.step_size,
                w.N, w.mean, w.var)
        f = open('saved_controllers/{}.pkl'.format(name), 'wb')
        pickle.dump(params, f)
        f.close()

    def load(self, name):
        f = open('saved_controllers/{}.pkl'.format(name), 'rb')
        params = pickle.load(f)
        self.w_policy = params[0]
        self.episodes = params[1]
        self.n_dirs = params[2]
        self.eps = params[3]
        self.step_size = params[4]
        self.whitener.N = params[5]
        self.whitener.mean = params[6]
        self.whitener.var = params[7]
        f.close()

    def demo(self):
        ret = self.whitener.rt(self.w_policy, seed=None, visual=True, stats=False, shift=0.0)
        print(self.episodes, ret)

if __name__ == "__main__":
    print("Making env")
    env = gym.make('DartHopper-v1')
    rs = RandomSearch(env, 8, 0.01, 0.05)
    print("Starting search")
    rs.random_search(25)
    #rs.random_search(250)
    embed()
