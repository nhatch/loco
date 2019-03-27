import cma
import numpy as np
import simbicon_params as sp
from IPython import embed

class CMAWrapper():
    def reset(self, initial_sigma=0.15, initial_cov=None):
        self.sigma = initial_sigma
        self.cov = initial_cov
        if self.cov is None:
            self.cov = np.diag(sp.PARAM_SCALE**2)

    def optimize(self, runner, initial_action, settings):
        cp = settings['controllable_params']
        def f(action, video_save_dir=None, render=None):
            new_action = initial_action.copy()
            new_action[cp] = action.reshape((-1,1))
            runner.reset(video_save_dir, render)
            return 1-runner.run(new_action.reshape(-1))
        self.initial_action = initial_action
        new_initial_action = initial_action[cp]
        initial_cov = self.cov[cp,:][:,cp]
        opzer = cma.CMA(f, new_initial_action, self.sigma, initial_cov)
        return self.do_iters(opzer, settings)

    def do_iters(self, opzer, settings):
        i = 0
        while True:
            val = opzer.f(opzer.mean, render=1)
            print('Iteration', i, ':', val)
            if val < settings['tol']:
                return self.finalize(opzer, settings)
            if i > settings['max_iters']:
                return None
            opzer.iter()
            i += 1

    def finalize(self, opzer, settings):
        cov = self.cov.copy()
        cp = settings['controllable_params'].reshape((-1,1))
        cov[cp.dot(cp.T)] = opzer.cov.reshape(-1)
        self.cov = cov
        self.sigma = opzer.sigma
        action = self.initial_action.copy()
        action[cp.reshape(-1)] = opzer.mean
        return action
