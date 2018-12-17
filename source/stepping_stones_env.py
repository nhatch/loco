import pydart2 as pydart
from IPython import embed
from enum import Enum
import numpy as np
import random_search

from simbicon import Simbicon
from pd_control import PDController
from sdf_loader import SDFLoader, RED, GREEN, BLUE
from video_recorder import video_recorder
from state import State
import consts_2D

# For rendering
from gym.envs.dart.static_window import *
from pydart2.gui.trackball import Trackball

SIMULATION_RATE = 1.0 / 2000.0 # seconds
EPISODE_TIME_LIMIT = 10.0 # seconds
REAL_TIME_STEPS_PER_RENDER = 25 # Number of simulation steps to run per frame so it looks like real time. Just a rough estimate.

class StepResult(Enum):
    ERROR = -1
    IN_PROGRESS = 0
    COMPLETE = 1

class SteppingStonesEnv:
    def __init__(self, controller_class=Simbicon):
        self.controller_class = controller_class
        self.controller = None
        self.world = None
        self.viewer = None
        self.sdf_loader = SDFLoader()
        pydart.init(verbose=False)
        self.clear_skeletons()

        self.video_recorder = None
        # Hacks to make this work with the gym.wrappers Monitor API
        self.metadata = {'render.modes': ['rgb_array', 'human']}
        #self.reward_range = range(10)
        #self.spec = None

        c = self.consts()
        c.observable_features = np.concatenate([
                c.observable_features_q,
                c.observable_features_q,
                c.observable_features_t,
                c.observable_features_t,
                ])
        self.is_3D = (c.BRICK_DOF == 6)

    def consts(self):
        return consts_2D

    def wrap_state(self, raw_state):
        return State(raw_state)

    def clear_skeletons(self):
        # pydart2 has not implemented any API to remove skeletons, so we need
        # to recreate the entire world.
        if self.world is not None:
            self.world.destroy()
        world = self.load_world()
        self.world = world
        self.sdf_loader.reset(world)
        walker = world.skeletons[1]
        assert(walker.name == "walker")
        # TODO I spent a whole day tracking down weird behavior that was ultimately due to an incorrect joint limit. Is there a way that I can visualize when these limits are being hit?
        for j in walker.joints:
            j.set_position_limit_enforced()
        self.robot_skeleton = walker

        if self.controller is None:
            self.controller = self.controller_class(walker, self)
        self.controller.skel = walker
        walker.set_controller(self.controller)
        self.controller.reset()

        if self.viewer is not None:
            self.viewer.sim = world
        else:
            title = None
            win = StaticGLUTWindow(self.world, title)
            # For some reason setting 'zoom' doesn't do anything
            win.scene.add_camera(self.init_camera(), 'gym_camera')
            win.scene.set_camera(win.scene.num_cameras()-1)
            win.run()
            self.viewer = win

    def init_camera(self):
        tb = Trackball(theta=-45.0, phi = 0.0)
        tb.trans[2] = -5
        return tb

    def reset(self, state=None, video_save_dir=None, random=0.0, seed=None):
        if seed is None:
            seed = np.random.randint(100000)
        print("Seed:", seed)
        np.random.seed(seed)

        self.world.reset()
        self.set_state(state, random)
        if self.video_recorder:
            self.video_recorder.close()
        if video_save_dir:
            self.video_recorder = video_recorder(self, video_save_dir)
        else:
            self.video_recorder = None
        return self.current_observation()

    def find_contacts(self, bodynode):
        return [c for c in self.world.collision_result.contacts if (c.bodynode1 == bodynode or c.bodynode2 == bodynode)]

    # Executes one world step.
    # Returns (observation, episode_terminated, human-readable message) tuple.
    def simulation_step(self):
        c = self.consts()
        swing_idx = self.controller.swing_idx
        swing_foot = self.controller.ik.get_bodynode(swing_idx, c.FOOT_BODYNODE_OFFSET)
        contacts = self.find_contacts(swing_foot)
        swing_heel = self.controller.ik.forward_kine(swing_idx)
        crashed = self.controller.crashed(swing_heel)
        step_complete = crashed or self.controller.swing_contact(contacts, swing_heel)
        if step_complete:
            status_string = self.controller.change_stance(contacts, swing_heel)

        obs = self.current_observation()
        if self.world.time() > EPISODE_TIME_LIMIT:
            return obs, True, "ERROR: Time limit reached"
        elif crashed or obs.crashed():
            return obs, True, "ERROR: Crashed"
        elif step_complete:
            return obs, False, status_string
        else:
            self.world.step()
            return None, False, None

    # Run one footstep of simulation, returning the final state
    def simulate(self, target, action=None, render=False, put_dots=False):
        self.controller.set_gait_raw(raw_gait=action, target=target)
        steps_per_render = None
        if render:
            steps_per_render = int(REAL_TIME_STEPS_PER_RENDER / render)
            if put_dots:
                self.sdf_loader.put_dot(target, 'step_target', color=GREEN)
                self.sdf_loader.put_dot(self.controller.prev_target, 'prev_step_target', color=GREEN)
        while True:
            if steps_per_render and self.world.frame % steps_per_render == 0:
                self._render()
            obs, terminated, status_string = self.simulation_step()
            if obs is not None:
                if render:
                    self._render()
                    self.log(status_string)
                return obs, terminated

    # Maps the raw agent state to a standardized ordering of DOFs with standardized signs.
    # The length of the array should not change. The inverse operation is `from_features`.
    def to_features(self, q):
        c = self.consts()
        q = q[c.perm]
        q[c.sign_switches] *= -1
        return q

    def get_x(self):
        q = self.to_features(np.array(self.robot_skeleton.q))
        dq = self.to_features(np.array(self.robot_skeleton.dq))
        return q, dq

    def from_features(self, q):
        c = self.consts()
        q = q.copy()
        q[c.sign_switches] *= -1
        base = np.zeros(c.Q_DIM)
        for i,j in enumerate(c.perm):
            base[j] = q[i]
        return base

    def current_observation(self):
        obs = np.concatenate(self.get_x())
        obs = np.concatenate((obs, self.controller.state()))
        if self.controller.swing_idx == self.consts().LEFT_IDX:
            obs = self.controller.standardize_stance(obs)
        return self.wrap_state(obs)

    def log(self, string):
        print(string)

    def trace(self, string, every=30):
        if self.world.frame % every == 0:
            self.log(string)

    def set_state(self, state, random):
        # If random = 0.0, no randomness will be added.
        ndofs = self.robot_skeleton.ndofs
        perturbation = np.random.uniform(low=-random, high=random, size=2*ndofs)
        if state is None:
            self.robot_skeleton.x += perturbation
            # Start with some forward momentum (Simbicon has some trouble otherwise)
            dq = np.zeros(ndofs)
            dq[0] += 0.4 + random * np.random.uniform(low=0.0, high=0.4)
            dq = self.robot_skeleton.dq + self.from_features(dq)
            self.robot_skeleton.dq = dq
            self.controller.reset()
        else:
            perturbation[:ndofs] += self.from_features(state.raw_state[:ndofs])
            perturbation[ndofs:] += self.from_features(state.raw_state[ndofs:2*ndofs])
            self.robot_skeleton.x = perturbation
            self.controller.reset(state)

    def _render(self):
        if self.video_recorder:
            self.video_recorder.capture_frame()
        else:
            self.render()

    def render(self, mode='human', close=False):
        self.update_viewer(self.robot_skeleton.com())
        if mode == 'rgb_array':
            data = self.viewer.getFrame()
            return data
        elif mode == 'human':
            self.viewer.runSingleStep()

    def update_viewer(self, com):
        self.viewer.scene.tb.trans[0] = -com[0]

    def gui(self):
        pydart.gui.viewer.launch(self.world)

    def load_world(self):
        skel = "skel/walker2d.skel"
        world = pydart.World(SIMULATION_RATE, skel)
        return world

