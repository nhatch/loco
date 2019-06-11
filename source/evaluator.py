import numpy as np
import utils
from sdf_loader import GREEN
from IPython import embed

class Evaluator:
    def __init__(self, env):
        self.env = env

    def set_eval_settings(self, settings):
        self.eval_settings = settings
        if self.eval_settings['use_stepping_stones']:
            self.env.sdf_loader.ground_width = self.eval_settings['ground_width']
            self.env.sdf_loader.ground_length = self.eval_settings['ground_length']
        else:
            self.env.sdf_loader.ground_length = 7.5
            self.env.sdf_loader.ground_width = self.env.consts().DEFAULT_GROUND_WIDTH
        self.env.clear_skeletons()

    def evaluate(self, policy, render=1.0, video_save_dir=None, seed=None, max_intolerable_steps=1):
        s = self.eval_settings
        seed = seed or np.random.randint(100000)
        state = self.env.reset(video_save_dir=video_save_dir, seed=seed, random=0.005, render=render)
        max_error = 0
        total_reward = 0
        DISCOUNT = 0.8
        targets = self.generate_targets(state)
        if s['use_stepping_stones']:
            self.env.sdf_loader.put_grounds(targets)
        else:
            self.env.sdf_loader.put_grounds(targets[:1])
            if render is not None:
                for i,t in enumerate(targets[2:]):
                    self.env.sdf_loader.put_dot(t, str(i), color=GREEN)
        total_offset = 0
        num_successful_steps = 0
        num_intolerable_steps = 0
        experience = []
        for i in range(s['n_steps']):
            target = targets[2+i]
            features = state.extract_features(target)
            action = policy(features)
            raw_pose_start = self.env.robot_skeleton.x
            end_state, terminated = self.env.simulate(target, target_heading=0.0, action=action)
            error = np.linalg.norm(end_state.stance_heel_location() - target)
            reward = utils.reward(self.env.controller, end_state)
            total_reward += reward
            # Actually `state` is just an observation. (In the "armless" model, for example,
            # it's missing info from two DOFs.) So we also include `raw_pose_start` so that
            # learning in simulation can accurately reset to the starting state.
            # In RL terms, (state,target,raw_pose_start) is the state.
            # To turn this into a vector suitable for ML processing, use
            # state.extract_features(target).
            debug_info = (seed, i)
            experience.append((state, target, raw_pose_start, action, reward, debug_info))
            if (max_intolerable_steps is not None) and (error > s['termination_tol']):
                num_intolerable_steps += 1
                terminate_early = (num_intolerable_steps >= max_intolerable_steps)
                terminated = terminated or terminate_early
            if error > max_error:
                max_error = error
            if terminated:
                break
            state = end_state
            num_successful_steps += 1
        if video_save_dir:
            self.env.close_video_recorder()
        result = {
                "total_reward": total_reward,
                "n_steps": num_successful_steps,
                "max_error": max_error,
                "seed": seed,
                }
        # We don't want to return this because it clutters up the output when
        # run in the interpreter. Access it via evaluator.experience.
        self.experience = experience
        return result

    def generate_targets(self, start_state):
        s = self.eval_settings
        targets = start_state.starting_platforms()
        current_swing_left = start_state.swing_left
        next_target = targets[-1]
        # TODO should we make the first step a bit shorter, since we're starting from "rest"?
        for i in range(s['n_steps']):
            dx = s['dist_mean'] + s['dist_spread'] * (np.random.uniform() - 0.5)
            dy = s['y_mean'] + s['y_spread'] * (np.random.uniform() - 0.5)
            dz = s['z_mean'] + s['z_spread'] * (np.random.uniform() - 0.5)
            if current_swing_left:
                dz *= -1
            current_swing_left = not current_swing_left
            next_target = next_target + [dx, dy, dz]
            targets.append(next_target)
        # Return value includes the two starting platforms
        return targets
