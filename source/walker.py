import pydart2 as pydart
from IPython import embed
from enum import Enum
import numpy as np
import random_search

from simbicon import SIMBICON_ACTION_SIZE
from pd_control import PDController
from sdf_loader import SDFLoader, RED, GREEN, BLUE
from video_recorder import video_recorder
from state import State

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

class TwoStepEnv:
    def __init__(self, controller_class):
        self.controller_class = controller_class
        self.world = None
        self.viewer = None
        self.sdf_loader = SDFLoader()
        self.clear_skeletons()

        # We just want this to be something that has a "shape" method
        # TODO incorporate adding target step locations into the observations
        self.observation_space = np.zeros_like(self.current_observation().raw_state)
        self.action_space = np.zeros(SIMBICON_ACTION_SIZE)

        self.video_recorder = None
        # Hacks to make this work with the gym.wrappers Monitor API
        self.metadata = {'render.modes': ['rgb_array', 'human']}
        #self.reward_range = range(10)
        #self.spec = None

    def clear_skeletons(self):
        # pydart2 has not implemented any API to remove skeletons, so we need
        # to recreate the entire world.
        if self.world is not None:
            self.world.destroy()
        world = load_world()
        self.world = world
        self.sdf_loader.reset(world)
        walker = world.skeletons[1]
        self.robot_skeleton = walker
        self.r_foot = walker.bodynodes[5]
        self.l_foot = walker.bodynodes[8]
        for j in walker.joints:
            j.set_position_limit_enforced()

        self.controller = self.controller_class(walker, self)
        walker.set_controller(self.controller)

        if self.viewer is not None:
            self.viewer.sim = world
        else:
            title = None
            win = StaticGLUTWindow(self.world, title)
            # For some reason setting 'zoom' doesn't do anything
            win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0), 'gym_camera')
            win.scene.set_camera(win.scene.num_cameras()-1)
            win.run()
            self.viewer = win

    def seed(self, seed):
        print("Seed:", seed)
        np.random.seed(seed)

    def reset(self, state=None, record_video=False, random=1.0):
        self.world.reset()
        self.set_state(state, random)
        if self.video_recorder:
            self.video_recorder.close()
        if record_video:
            self.video_recorder = video_recorder(self)
        else:
            self.video_recorder = None
        return self.current_observation()

    def find_contacts(self, bodynode):
        return [c for c in self.world.collision_result.contacts if c.bodynode1 == bodynode]

    # Executes one world step.
    # Returns (observation, episode_terminated, human-readable message) tuple.
    def simulation_step(self):
        swing_foot = self.robot_skeleton.bodynodes[self.controller.swing_idx+2]
        contacts = self.find_contacts(swing_foot)
        swing_heel = self.controller.ik.forward_kine(self.controller.swing_idx)[:2]
        step_complete = self.controller.step_complete(contacts, swing_heel)
        if step_complete:
            status_string = self.controller.change_stance(contacts, swing_heel)

        obs = self.current_observation()
        if self.world.time() > EPISODE_TIME_LIMIT:
            return obs, True, "ERROR: Time limit reached"
        elif obs.crashed():
            return obs, True, "ERROR: Crashed"
        elif step_complete:
            return obs, False, status_string
        else:
            self.world.step()
            return None, False, None

    # Run one footstep of simulation, returning the final state and the achieved step distance
    def simulate(self, target, action=None, render=False, put_dots=False):
        self.controller.set_gait_raw(raw_gait=action, target=target)
        steps_per_render = None
        if render:
            steps_per_render = int(REAL_TIME_STEPS_PER_RENDER / render)
            if put_dots:
                self.sdf_loader.put_dot(target, color=GREEN)
        while True:
            if steps_per_render and self.world.frame % steps_per_render == 0:
                self._render()
            obs, terminated, status_string = self.simulation_step()
            if obs is not None:
                if render:
                    self.log(status_string)
                return obs, terminated

    def current_observation(self):
        obs = self.robot_skeleton.x.copy()
        obs = self.standardize_stance(obs)
        return State(np.concatenate((obs, self.controller.state())))

    def standardize_stance(self, state):
        # Ensure the stance state is contained in state[6:9] and state[15:18].
        # Do this by flipping the left state with the right state if necessary.
        stance_idx, swing_idx = self.controller.stance_idx, self.controller.swing_idx
        stance_q = state[stance_idx:stance_idx+3].copy()
        stance_dq = state[stance_idx+9:stance_idx+12].copy()
        swing_q = state[swing_idx:swing_idx+3].copy()
        swing_dq = state[swing_idx+9:swing_idx+12].copy()
        state[3:6] = swing_q
        state[6:9] = stance_q
        state[12:15] = swing_dq
        state[15:18] = stance_dq
        return state

    def log(self, string):
        print(string)

    def set_state(self, state, random):
        # If random = 0.0, no randomness will be added
        if state is None:
            self.robot_skeleton.q += random * np.random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            # Start with some forward momentum (Simbicon has some trouble otherwise)
            dq = self.robot_skeleton.dq
            dq[0] += 0.8 + random * np.random.uniform(low=0.0, high=0.4)
            self.robot_skeleton.dq = dq
            self.controller.reset()
        else:
            self.robot_skeleton.x = state.pose()
            self.controller.reset(state.controller_state())

    def _render(self):
        if self.video_recorder:
            self.video_recorder.capture_frame()
        else:
            self.render()

    def render(self, mode='human', close=False):
        self.viewer.scene.tb.trans[0] = -self.robot_skeleton.com()[0]*1
        self.viewer.scene.tb.trans[2] = -5 # adjust zoom
        if mode == 'rgb_array':
            data = self.viewer.getFrame()
            return data
        elif mode == 'human':
            self.viewer.runSingleStep()

    def gui(self):
        pydart.gui.viewer.launch(self.world)

def load_world():
    skel = "skel/walker2d.skel"
    pydart.init(verbose=False)
    world = pydart.World(SIMULATION_RATE, skel)
    return world

