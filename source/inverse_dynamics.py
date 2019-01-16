from IPython import embed
import numpy as np
import pickle
import os

from sklearn.linear_model import LinearRegression, Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from simbicon import Simbicon
from simbicon_params import *
from state import State
from step_learner import Runner
from random_search import RandomSearch

N_STATES_PER_ITER = 16
START_STATES_FMT = 'data/start_states_{}.pkl'
TRAIN_FMT = 'data/train_{}.pkl'

RIDGE_ALPHA = 0.1

class LearnInverseDynamics:
    def __init__(self, env, exp_name=''):
        self.env = env
        self.exp_name = exp_name
        # Evaluation settings
        self.eval_settings = {
                'use_stepping_stones': True,
                'dist_mean': 0.47,
                'dist_spread': 0.3,
                'runway_length': 0.4,
                'n_steps': 16,
                'z_mean': 0.0,
                'z_spread': 0.0,
                'y_mean': 0.05,
                'y_spread': 0.1,
                }
        if self.env.is_3D:
            self.eval_settings = {
                'use_stepping_stones': False,
                'dist_mean': 0.35,
                'dist_spread': 0.2,
                'runway_length': 10.0,
                'n_steps': 16,
                'z_mean': 0.4,
                'z_spread': 0.1,
                'y_mean': 0.0,
                'y_spread': 0.0,
                }
        self.initialize_start_states()
        self.train_features, self.train_responses = [], []
        self.n_action = len(self.env.controller.action_params())
        self.n_dynamic = sum(self.env.consts().observable_features)
        # TODO try some model more complicated than linear?
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        #model = LinearRegression(fit_intercept=False)
        #model = make_pipeline(PolynomialFeatures(2), model) # quadratic
        #model = RANSACRegressor(base_estimator=model, residual_threshold=2.0)
        self.model = model
        self.is_fitted = False # For some dang reason sklearn doesn't track this itself
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

    def initialize_start_states(self):
        fname = START_STATES_FMT.format(self.exp_name)
        if not os.path.exists(fname):
            if self.env.is_3D:
                states = self.collect_starting_states(min_length=0.1, max_length=0.5)
            else:
                states = self.collect_starting_states()
            with open(fname, 'wb') as f:
                pickle.dump(states, f)
        with open(fname, 'rb') as f:
            self.start_states = pickle.load(f)

    def collect_starting_states(self, size=8, n_resets=8, min_length=0.2, max_length=0.8):
        self.env.log("Collecting initial starting states")
        start_states = []
        for i in range(n_resets):
            length = min_length + (max_length - min_length) * (i / n_resets)
            self.env.log("Starting trajectory {}".format(i))
            start_state = self.env.reset(render=0.1, random=0.005)
            self.env.sdf_loader.put_grounds([start_state.swing_platform()], runway_length=15)
            # TODO should we include this first state? It will be very different from the rest.
            start_states.append(self.env.current_observation())
            targets = self.generate_targets(start_state, size, dist_mean=length, dist_spread=0)
            for target in targets[2:]:
                # This makes the first step half as long. TODO is this necessary/sufficient?
                target[0] -= length*0.5
                end_state, _ = self.env.simulate(target, target_heading=0.0, put_dots=True)
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
        mean_action = self.learn_action(start_state, target)
        if mean_action is None:
            # Random search couldn't find a good enough action; don't use this for training.
            return
        self.append_to_train_set(start_state, target, mean_action)
        # The following assumes env hasn't changed since the last rs.eval() run.
        end_state = self.env.current_observation()
        self.start_states.append(end_state)

    def learn_action(self, start_state, target):
        runner = Runner(self.env, start_state, target)
        if env.is_3D:
            rs = RandomSearch(runner, 8, step_size=0.1, eps=0.1)
        else:
            rs = RandomSearch(runner, 4, step_size=0.1, eps=0.1)
        runner.reset()
        rs.w_policy = self.act(target) # Initialize with something reasonable
        # TODO put max_iters and tol in the object initialization params instead
        w_policy = rs.random_search(max_iters=5, tol=0.02, render=1)
        return w_policy

    def append_to_train_set(self, start_state, target, action):
        # The train set is a set of (features, action) pairs
        # where taking `action` when the environment features are `features`
        # will hit the target with the swing foot (within 5 cm).
        train_state = self.extract_features(start_state, target)
        self.train_features.append(train_state)
        self.train_responses.append(action)

    def extract_features(self, state, target):
        # Combines state and target into a single vector, and discards some information
        # that does not affect the dynamics (such as absolute location).
        # That this vector is still very high dimenstional for debugging purposes.
        # When actually training the model, many of these features are discarded.
        c = self.env.consts()
        centered_state = np.zeros(2*c.Q_DIM + 3*3)

        # Recenter all (x,y,z) coordinates around the location of the starting platform.
        # TODO: also rotate so absolute yaw is 0.
        base = state.stance_platform()
        centered_state[ 0:-9] = state.pose()
        centered_state[ 0: 3] -= base # starting location of agent
        centered_state[-9:-6] = state.stance_heel_location() - base
        centered_state[-6:-3] = state.swing_platform() - base
        centered_state[-3:  ] = target - base
        return centered_state

    def reconstruct_state(self, features):
        # At the moment, used only for debugging.
        raw_state = features.copy()
        raw_state[-3:  ] = features[-6:-3]
        raw_state[-6:-3] = [0,0,0]
        target = features[-3:].copy()
        if self.env.is_3D:
            c = self.env.consts()
            raw_state[[c.Y, -2, -5, -8]] -= 0.9
            target[c.Y] -= 0.9
        return State(raw_state), target

    def train_inverse_dynamics(self):
        c = self.env.consts()
        # TODO investigate removing more features to reduce problem dimension
        X = np.array(self.train_features)[:,c.observable_features]
        y = np.array(self.train_responses)
        self.X_mean = X.mean(0)
        self.y_mean = y.mean(0)
        X = (X - self.X_mean) / self.X_scale_factor
        y = (y - self.y_mean) * self.y_scale_factor
        self.model.fit(X, y)
        self.is_fitted = True

    def act(self, target):
        c = self.env.consts()
        state = self.env.current_observation()
        if self.env.controller.swing_idx == c.LEFT_IDX:
            # `state` has been mirrored, but `target` has not.
            target = target * np.array([1,1,-1])
        X = self.extract_features(state, target).reshape(1,-1)[:,c.observable_features]
        X = (X - self.X_mean) / self.X_scale_factor
        if self.is_fitted:
            action = self.model.predict(X).reshape(-1)
        else:
            action = np.zeros(self.n_action)
        # Simbicon3D will handle mirroring the action if necessary
        return action / self.y_scale_factor + self.y_mean

    def generate_targets(self, start_state, num_steps, dist_mean=None, dist_spread=None):
        s = self.eval_settings
        if dist_mean is None:
            dist_mean = s['dist_mean']
        if dist_spread is None:
            dist_spread = s['dist_spread']
        # Choose some random targets that are hopefully representative of what
        # we will see during testing episodes.
        targets = start_state.starting_platforms()
        next_target = targets[-1]
        for i in range(num_steps):
            dx = dist_mean + dist_spread * (np.random.uniform() - 0.5)
            dy = s['y_mean'] + s['y_spread'] * (np.random.uniform() - 0.5)
            dz = s['z_mean'] + s['z_spread'] * (np.random.uniform() - 0.5)
            if i % 2 == 1:
                dz *= -1
            next_target = next_target + [dx, dy, dz]
            targets.append(next_target)
        # Return value includes the two starting platforms
        return targets

    def training_iter(self):
        self.collect_dataset()
        self.train_inverse_dynamics()

    def evaluate(self, render=1.0, video_save_dir=None, seed=None):
        s = self.eval_settings
        state = self.env.reset(video_save_dir=video_save_dir, seed=seed, random=0.005, render=render)
        total_error = 0
        max_error = 0
        total_score = 0
        DISCOUNT = 0.8
        targets = self.generate_targets(state, s['n_steps'])
        if s['use_stepping_stones']:
            self.env.sdf_loader.put_grounds(targets, runway_length=s['runway_length'])
        else:
            self.env.sdf_loader.put_grounds(targets[:1], runway_length=s['runway_length'])
        total_offset = 0
        num_successful_steps = 0
        for i in range(s['n_steps']):
            target = targets[2+i]
            action = self.act(target)
            state, terminated = self.env.simulate(target, target_heading=0.0, action=action)
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

def retrieve_index(learn, i=None):
    learn.env.clear_skeletons()
    if i is None:
        i = np.random.randint(len(learn.train_features))
        print("Testing index:", i)
    features = learn.train_features[i]
    response = learn.train_responses[i]
    start_state, target = learn.reconstruct_state(features)
    return start_state, target, response

def demo_train_set(learn):
    for i in range(len(learn.train_responses)):
        start_state, target, response = retrieve_index(learn, i)
        runner = Runner(learn.env, start_state, target)
        runner.reset(render=1.0)
        print("Score:", runner.run(response)) # Should be pretty close to zero.

def test_regression_bias(learn, i=None):
    start_state, target, response = retrieve_index(learn, i)

    runner = Runner(learn.env, start_state, target)
    runner.reset(render=1.0)
    print("Score:", runner.run(response)) # Should be pretty close to zero.

    runner.reset(render=1.0)
    trained_response = learn.act(target)
    # This score should also be close to zero, depending on model bias.
    print("Score:", runner.run(trained_response))

    # TODO: It seems like the algorithm is almost (not completely) ignoring the target Z coord.
    target[2] += 0.2
    runner = Runner(learn.env, start_state, target)
    runner.reset(render=1.0)
    trained_response = learn.act(target)
    # This score should also be close to zero in the absence of model bias.
    print("Score:", runner.run(trained_response))

def test_mirroring(learn, i=None):
    start_state, target, response = retrieve_index(learn, i)
    runner = Runner(learn.env, start_state, target)
    trained_response = learn.act(target)
    runner.reset(render=1.0)
    print("Score:", runner.run(trained_response)) # Should be pretty close to zero.

    mirrored_start = State(learn.env.controller.mirror_state(start_state.raw_state.copy()))
    mirrored_target = target*[1,1,-1]
    mirrored_runner = Runner(learn.env, mirrored_start, mirrored_target)
    mirrored_runner.reset(render=1.0)
    learn.env.controller.change_stance([], mirrored_start.stance_heel_location())
    obs_after_mirror = learn.env.current_observation()
    response_after_mirror = learn.act(target)
    print(np.allclose(obs_after_mirror.raw_state, start_state.raw_state))
    print(np.allclose(response_after_mirror, trained_response))
    # This score should be identical to the previous score (actually about 1e-6 error)
    print("Score:", mirrored_runner.run(trained_response))

    # These scores should also be identical (actually about 1e-6 error)
    runner.reset(render=1.0)
    print("Score:", runner.run(response))
    mirrored_runner.reset(render=1.0)
    learn.env.controller.change_stance([], mirrored_start.stance_heel_location())
    print("Score:", mirrored_runner.run(response))

def hard_mode(learn):
    learn.eval_settings['use_stepping_stones'] = True
    env.sdf_loader.ground_width = 0.2
    env.sdf_loader.ground_length = 0.3
    # The concept of a "runway" in 3D is not properly implemented anyway. TODO?
    learn.eval_settings['runway_length'] = 0.3
    learn.eval_settings['dist_spread'] = 0.5
    learn.eval_settings['z_spread'] = 0.2

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    #env = SteppingStonesEnv()
    env = Simple3DEnv(Simbicon3D)

    name = '3D' if env.is_3D else '2D'
    name = 'foo'
    learn = LearnInverseDynamics(env, name)
    learn.load_train_set()
    #learn.training_iter()
    #print(learn.evaluate())
    #test_regression_bias(learn)
    test_mirroring(learn)
    embed()
