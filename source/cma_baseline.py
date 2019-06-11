from IPython import embed
import numpy as np
import cma

from state import reconstruct_state
import curriculum as cur
from step_learner import Runner
from rs_baseline import coef_to_fn, RandomSearchBaseline

class CMABaseline(RandomSearchBaseline):
    def flatify(self, coef_):
        cp = self.train_settings['controllable_params']
        return coef_[cp].reshape(-1)

    def squarify(self, coef):
        cp = self.train_settings['controllable_params']
        full_coef = np.zeros_like(cp, dtype=coef.dtype)
        full_coef[cp] = coef
        return full_coef

    def set_train_settings(self, settings):
        super().set_train_settings(settings)
        cp = self.train_settings['controllable_params']
        self.opzer = cma.CMA(None, self.flatify(self.coef_), 0.01, np.eye(cp.sum()))
        # TODO how to save the opzer state for checkpointing experiments?
        # Currently each call to set_train_setting overwrites all of that state
        # (except for the mean).

    def _iter(self):
        result = self.evaluate(render=None)
        self.env.clear_skeletons()
        experience = self.evaluator.experience
        cp = self.train_settings['controllable_params']
        runners = []
        for start_state, target, raw_pose_start, _, _, _ in experience:
            features = start_state.extract_features(target)
            runner = Runner(self.env, start_state, target, raw_pose_start=raw_pose_start)
            runners.append((features, runner))
        def f(coef, video_save_dir=None, render=None):
            full_coef = self.squarify(coef)
            total_reward = 0.0
            for features, runner in runners:
                action = coef_to_fn(full_coef)(features)
                runner.reset(video_save_dir, 1)
                total_reward += 1-runner.run(action.reshape(-1))
            return total_reward / len(experience)
        n_iters = 1
        self.opzer.f = f
        for i in range(n_iters):
            self.opzer.iter()
        for _, runner in runners:
            self.total_steps += runner.n_runs
        self.coef_ = self.squarify(self.opzer.mean.reshape(-1))
        self.env.clear_skeletons()

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
