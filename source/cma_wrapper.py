import cma

class CMAWrapper():
    def __init__(self, f, initial_mean, initial_sigma, settings):
        self.settings = settings
        self.initial_mean = initial_mean
        def new_f(x, video_save_dir=None, render=None):
            new_x = initial_mean.copy()
            new_x[settings['controllable_params']] = x.reshape((-1,1))
            return f(new_x, video_save_dir, render)
        new_initial_mean = initial_mean[settings['controllable_params']]
        self.opzer = cma.CMA(new_f, new_initial_mean, initial_sigma)

    def optimize(self):
        i = 0
        while True:
            val = self.opzer.f(self.opzer.mean, render=1)
            print('Iteration', i, ':', val)
            if val < self.settings['tol']:
                r = self.initial_mean.copy()
                r[self.settings['controllable_params']] = self.opzer.mean
                return r
            if i > self.settings['max_iters']:
                return None
            self.opzer.iter()
            i += 1
