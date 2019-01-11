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

N_ACTIONS_PER_STATE = 1
N_STATES_PER_ITER = 32
EXPLORATION_STD = 0.0 # Don't do perturbations until we're sure it actually helps
START_STATES_FMT = 'data/start_states_{}.pkl'
TRAIN_FMT = 'data/train_{}.pkl'

RIDGE_ALPHA = 0.1

class LearnInverseDynamics:
    def __init__(self, env, exp_name=''):
        self.env = env
        self.exp_name = exp_name
        self.video_save_dir = None
        self.initialize_start_states()
        self.train_features, self.train_responses = [], []
        self.n_action = len(self.env.controller.action_params())
        self.n_dynamic = sum(self.env.consts().observable_features)
        # TODO try some model more complicated than linear?
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
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

        # Evaluation settings
        self.dist_mean = 0.42
        self.dist_spread = 0.0 if env.is_3D else 0.3
        self.n_steps = 16
        self.runway_length = 0.4

    def initialize_start_states(self):
        fname = START_STATES_FMT.format(self.exp_name)
        if not os.path.exists(fname):
            if env.is_3D:
                states = self.collect_starting_states(min_length=0.1, max_length=0.7)
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
            # TODO should we perturb the start state and target, too?
            runner.reset()
            # TODO test whether these perturbations actually help
            perturbation = EXPLORATION_STD * np.random.randn(len(mean_action))
            action = mean_action + perturbation
            end_state, terminated = self.env.simulate(target, action=action)
            if not terminated:
                self.append_to_train_set(start_state, target, action, end_state)
                self.start_states.append(end_state)

    def learn_action(self, start_state, target):
        runner = Runner(self.env, start_state, target)
        runner.video_save_dir = self.video_save_dir
        rs = RandomSearch(runner, 4, step_size=0.1, eps=0.05)
        rs.w_policy = self.act(start_state, target) # Initialize with something reasonable
        # TODO put max_iters and tol in the object initialization params instead
        w_policy = rs.random_search(max_iters=10, tol=0.05, render=1)
        return w_policy, runner

    def append_to_train_set(self, start_state, target, action, end_state):
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
        target = features[-3:]
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

    def act(self, state, target, flip_z=False):
        # `state` has been standardized, but `target` has not.
        # TODO there must be a cleaner way to handle this.
        m = np.array([1,1,-1]) if flip_z else np.array([1,1,1])
        c = self.env.consts()
        X = self.extract_features(state, target*m).reshape(1,-1)[:,c.observable_features]
        X = (X - self.X_mean) / self.X_scale_factor
        if self.is_fitted:
            action = self.model.predict(X).reshape(-1)
        else:
            action = np.zeros(self.n_action)
        return action / self.y_scale_factor + self.y_mean

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
            state, terminated = self.env.simulate(target, action=action, render=render)
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

def test_train_state_storage(learn, i=None):
    learn.env.clear_skeletons()
    if i is None:
        i = np.random.randint(len(learn.train_features))
        print("Testing index:", i)
    features = learn.train_features[i]
    response = learn.train_responses[i]
    start_state, target = learn.reconstruct_state(features)
    runner = Runner(learn.env, start_state, target)
    score = runner.run(response, render=1.0)
    # This score should be pretty close to zero.
    trained_response = learn.act(start_state, target)
    runner.run(trained_response, render=1.0)
    # This score should also be close to zero in the absence of model bias.

if __name__ == '__main__':
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    env = SteppingStonesEnv()
    #env = Simple3DEnv(Simbicon3D)

    name = '3D' if env.is_3D else '2D'
    learn = LearnInverseDynamics(env, name)
    #learn.video_save_dir = 'monitoring'
    learn.load_train_set()
    learn.training_iter()
    #learn.evaluate()
    #test_train_state_storage(learn)
    embed()
