from IPython import embed
import numpy as np
import pickle
import os

from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from simbicon import Simbicon
from simbicon_params import *
from state import State
from step_learner import Runner
from random_search import RandomSearch

N_ACTIONS_PER_STATE = 4
N_STATES_PER_ITER = 32
EXPLORATION_STD = 0.1
START_STATES_FMT = 'data/start_states_{}.pkl'
TRAIN_FMT = 'data/train_{}.pkl'

RIDGE_ALPHA = 10.0

class LearnInverseDynamics:
    def __init__(self, env, exp_name=''):
        self.env = env
        self.exp_name = exp_name
        self.initialize_start_states()
        self.train_features, self.train_responses = [], []
        self.n_action = len(self.env.controller.base_gait())
        self.n_dynamic = sum(self.env.consts().observable_features)
        # TODO try some model more complicated than linear?
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        #model = make_pipeline(PolynomialFeatures(2), model) # quadratic
        self.model = RANSACRegressor(base_estimator=model, residual_threshold=2.0)
        self.X_mean = np.zeros(self.n_dynamic)
        self.y_mean = np.zeros(self.n_action)
        # Maybe try varying these.
        # Increasing X factor increases the penalty for using that feature as a predictor.
        # Increasing y factor increases the penalty for mispredicting that action parameter.
        # Originally I tried scaling by the variance, but that led to very poor performance.
        # It appears that variance in whatever arbitrary units is not a good indicator of
        # importance in fitting a linear model of the dynamics for this environment.
        self.X_scale_factor = np.ones(self.n_dynamic)
        self.y_scale_factor = np.ones(self.n_action)

        # Evaluation settings
        self.dist_mean = 0.42
        self.dist_spread = 0.0 if env.is_3D else 0.3
        self.n_steps = 16
        self.runway_length = 0.4

    def initialize_start_states(self):
        fname = START_STATES_FMT.format(self.exp_name)
        if not os.path.exists(fname):
            if env.is_3D:
                states = self.collect_starting_states(n_resets=8, min_length=0.1, max_length=0.7)
            else:
                states = self.collect_starting_states()
            with open(fname, 'wb') as f:
                pickle.dump(states, f)
        with open(fname, 'rb') as f:
            self.start_states = pickle.load(f)

    def collect_starting_states(self, size=8, n_resets=16, min_length=0.2, max_length=0.8):
        self.env.log("Collecting initial starting states")
        start_states = []
        for i in range(n_resets):
            length = min_length + (max_length - min_length) * (i / n_resets)
            self.env.log("Starting trajectory {}".format(i))
            start_state = self.env.reset(random=0.005)
            self.env.sdf_loader.put_grounds([start_state.swing_platform()], runway_length=15)
            # TODO should we include this first state? It will be very different from the rest.
            #start_states.append(self.env.robot_skeleton.x)
            targets = self.generate_targets(start_state, size, dist_mean=length, dist_spread=0)
            for target in targets[2:]:
                # This makes the first step half as long. TODO is this necessary/sufficient?
                target[0] -= length*0.5
                end_state, _ = self.env.simulate(target, render=0.1, put_dots=True)
                # Fix end_state.starting_platforms to reflect where the swing foot
                # actually ended up. We must do this because stance_platform and
                # stance_heel_location might actually be quite far apart, since there's
                # no optimization procedure here to guarantee otherwise.
                # TODO should we just make this a "flat runway" part of the curriculum?
                # Don't change the Y coordinate, since we want this all to be flat ground.
                end_state.stance_platform()[[0,2]] = end_state.stance_heel_location()[[0,2]]
                start_states.append(end_state)
        self.env.clear_skeletons()
        return start_states

    def dump_train_set(self):
        fname = TRAIN_FMT.format(self.exp_name)
        with open(fname, 'wb') as f:
            pickle.dump((self.start_states, self.train_features, self.train_responses), f)

    def load_train_set(self):
        fname = TRAIN_FMT.format(self.exp_name)
        with open(fname, 'rb') as f:
            self.start_states, self.train_features, self.train_responses = pickle.load(f)
        self.train_inverse_dynamics()

    def collect_dataset(self):
        self.env.clear_skeletons()
        indices = np.random.choice(range(len(self.start_states)), size=N_STATES_PER_ITER)
        for i,j in enumerate(indices):
            print("Exploring start state {} ({} / {})".format(j, i, len(indices)))
            self.collect_samples(self.start_states[j])
        print(len(self.start_states), len(self.train_features))
        self.dump_train_set()
        self.env.clear_skeletons()

    def collect_samples(self, start_state):
        # We don't need to flip this target, because start_state is standardized.
        target = self.generate_targets(start_state, 1)[-1]
        mean_action, runner = self.learn_action(start_state, target)
        if mean_action is None:
            # Random search couldn't find a good enough action; don't use this for training.
            return
        for _ in range(N_ACTIONS_PER_STATE):
            runner.reset()
            # TODO test whether these perturbations actually help
            perturbation = EXPLORATION_STD * np.random.randn(len(mean_action))
            perturbation *= self.env.controller.controllable_indices()
            action = mean_action + perturbation
            end_state, terminated = self.env.simulate(target, action)
            if not terminated:
                self.append_to_train_set(start_state, target, action, end_state)
                self.start_states.append(end_state)

    def learn_action(self, start_state, target):
        runner = Runner(self.env, start_state, target)
        rs = RandomSearch(runner, 4, step_size=0.1, eps=0.05)
        rs.video_save_dir = self.video_save_dir
        render = 0.7 if self.video_save_dir else 1.0
        rs.w_policy = self.act(start_state, target) # Initialize with something reasonable
        # TODO put max_iters and tol in the object initialization params instead
        w_policy = rs.random_search(max_iters=10, tol=0.05, render=render)
        return w_policy, runner

    def append_to_train_set(self, start_state, target, action, end_state):
        # The train set is a set of (features, action) pairs
        # where taking `action` when the environment features are `features`
        # will *exactly* hit the target with the swing foot.
        # Note that the target must be part of `features` and that
        # the `features` must capture all information necessary to reconstruct the
        # world state to verify that this action will indeed hit the target.
        # But it also must be reasonably low-dimensional because these are also the
        # features used in our ML algorithm. TODO: separate these two responsibilities.
        achieved_target = end_state.stance_heel_location()
        # TODO: This "hindsight retargeting" doesn't exactly work. The problem is that
        # moving the target platform to the new location actually changes end-of-step
        # detection (e.g. the toe hits something where before there was empty space).
        action[TX:TX+3] += target - achieved_target
        train_state = self.extract_features(start_state, achieved_target)
        self.train_features.append(train_state)
        self.train_responses.append(action)

    def extract_features(self, state, target):
        c = self.env.consts()
        centered_state = np.zeros(2*c.Q_DIM + 2*3)

        # Absolute location does not affect dynamics, so recenter all (x,y) coordinates
        # around the location of the starting platform.
        # TODO: also rotate so absolute yaw is 0. TODO this will also require doing the same
        # to the action that was taken?
        base = state.stance_platform()
        centered_state[ 0:-6] = state.pose()
        centered_state[ 0: 3] -= base # starting location of agent
        centered_state[-6:-3] = state.stance_heel_location() - base
        # We don't include state.swing_platform or state.stance_platform
        # because they seem less important
        # (and we are desperate to reduce problem dimension).
        centered_state[-3:  ] = target - base

        return centered_state[c.observable_features]

    def train_inverse_dynamics(self):
        X = np.array(self.train_features)
        y = np.array(self.train_responses)
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean) / self.X_scale_factor
        y = (y - self.y_mean) * self.y_scale_factor
        self.model.fit(X, y)

    def act(self, state, target, flip_z=False):
        # `state` has been standardized, but `target` has not.
        # TODO there must be a cleaner way to handle this.
        m = np.array([1,1,-1]) if flip_z else np.array([1,1,1])
        X = self.extract_features(state, target*m).reshape(1,-1)
        X = (X - self.X_mean) / self.X_scale_factor
        if hasattr(self.model, 'estimator_'):
            rescaled_action = self.model.predict(X).reshape(-1)
        else:
            rescaled_action = np.zeros(self.n_action)
        rescaled_action *= self.env.controller.controllable_indices()
        if flip_z:
            rescaled_action[FLIP_Z] *= -1
        return rescaled_action / self.y_scale_factor + self.y_mean

    def generate_targets(self, start_state, num_steps, dist_mean=None, dist_spread=None):
        if dist_mean is None:
            dist_mean = self.dist_mean
        if dist_spread is None:
            dist_spread = self.dist_spread
        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        targets = start_state.starting_platforms()
        next_target = targets[-1]
        for i in range(num_steps):
            dx = dist_mean + dist_spread * (np.random.uniform() - 0.5)
            dy = np.random.uniform() * 0.0
            # TODO maybe off by -1
            dz = 0.3 * (1 if i % 2 == 0 else -1) * (1 if self.env.is_3D else 0)
            next_target = next_target + [dx, dy, dz]
            targets.append(next_target)
        # Return value includes the two starting platforms
        return targets

    def training_iter(self):
        self.collect_dataset()
        self.train_inverse_dynamics()

    def evaluate(self, render=1.0, video_save_dir=None, seed=None):
        state = self.env.reset(video_save_dir=video_save_dir, seed=seed, random=0.005)
        total_error = 0
        max_error = 0
        total_score = 0
        DISCOUNT = 0.8
        targets = self.generate_targets(state, self.n_steps)
        self.env.sdf_loader.put_grounds(targets, runway_length=self.runway_length)
        total_offset = 0
        num_successful_steps = 0
        for i in range(self.n_steps):
            target = targets[2+i]
            flip_z = self.env.is_3D and (i % 2 == 1)
            action = self.act(state, target, flip_z=flip_z)
            state, terminated = self.env.simulate(target, action, render=render)
            if terminated:
                break
            num_successful_steps += 1
            achieved_target = state.stance_heel_location()
            error = np.linalg.norm(achieved_target - target)
            if error > max_error:
                max_error = error
            total_error += error
            if error < 1:
                total_score += (1-error) * (DISCOUNT**i)
        if video_save_dir:
            self.env.reset() # This ensures the video recorder is closed properly.
        result = {
                "total_score": total_score,
                "n_steps": num_successful_steps,
                "max_error": max_error,
                "total_error": total_error,
                }
        return result

def test_train_state_storage(env):
    learn = LearnInverseDynamics(env)
    start_state = np.random.choice(learn.start_states)
    learn.collect_samples(start_state)
    for i in range(len(learn.train_features)):
        features = learn.train_features[i]
        response = learn.train_responses[i]
        target = start_state.stance_platform().copy()
        target[0:2] += features[-2:]
        # TODO reconstruct start_state from features (in order to see things from the
        # perspective of the learning algorithm)
        runner = Runner(env, start_state, target)
        score = runner.run(response, render=1.0)
        print("Score:", score) # This score should be exactly zero.
        # But often it isn't; see todo in append_to_train_set

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    #env = SteppingStonesEnv()
    env = Simple3DEnv(Simbicon3D)
    #test_train_state_storage(env)

    name = '3D' if env.is_3D else '2D'
    learn = LearnInverseDynamics(env, name)
    learn.video_save_dir = None
    #learn.load_train_set()
    learn.training_iter()
    embed()
