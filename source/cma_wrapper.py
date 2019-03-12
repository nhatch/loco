import cma
import numpy as np
import simbicon_params as sp

class CMAWrapper():
    def __init__(self, initial_sigma=0.05, initial_cov=None):
        self.sigma = initial_sigma
        self.cov = initial_cov
        if self.cov is None:
            self.cov = np.diag(sp.PARAM_SCALE)

    def optimize(self, f, initial_mean, settings):
        cp = settings['controllable_params']
        def new_f(x, video_save_dir=None, render=None):
            new_x = initial_mean.copy()
            new_x[cp] = x.reshape((-1,1))
            return f(new_x, video_save_dir, render)
        self.initial_mean = initial_mean
        new_initial_mean = initial_mean[cp]
        initial_cov = self.cov[cp,:][:,cp]
        opzer = cma.CMA(new_f, new_initial_mean, self.sigma, initial_cov)
        self.do_iters(opzer, settings)

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
        r = self.initial_mean.copy()
        r[cp.reshape(-1)] = opzer.mean
        return r
