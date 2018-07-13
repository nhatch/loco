import numpy as np
from IPython import embed
import pickle

class Whitener:
    def __init__(self, env):
        self.N = 0
        self.env = env
        obs_dim = env.observation_space.shape[0]
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)

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
