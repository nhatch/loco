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
import time

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
        self.sdf_loader = SDFLoader(self.consts().DEFAULT_GROUND_WIDTH)
        pydart.init(verbose=False)
        c = self.consts()
        self.is_3D = (c.BRICK_DOF == 6)
        self.clear_skeletons()

        self.video_recorder = None
        # Hacks to make this work with the gym.wrappers Monitor API
        self.metadata = {'render.modes': ['rgb_array', 'human']}
        #self.reward_range = range(10)
        #self.spec = None

    def consts(self):
        return consts_2D

    def wrap_state(self, raw_state):
        swing_left = self.controller.swing_idx == self.consts().LEFT_IDX
        return State(raw_state, swing_left, self.consts())

    def clear_skeletons(self):
        # pydart2 has not implemented any API to remove skeletons, so we need
        # to recreate the entire world.
        if self.world is not None:
            self.world.destroy()
        world = self.load_world()
        self.world = world
        self.sdf_loader.reset(world)

        if self.controller is None:
            self.controller = self.controller_class(self)
        self.robot_skeleton.set_controller(self.controller)
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
        self.track_point = None
        self.zoom = 5.0
        def zoom_to(dx, dy):
            self.zoom -= dy/20
        tb.zoom_to = zoom_to
        return tb

    def reset(self, state=None, video_save_dir=None, render=None, random=0.0, seed=None):
        if seed is None:
            seed = np.random.randint(100000)
        np.random.seed(seed)

        self.world.reset()
        self.set_state(state, random)
        if self.video_recorder:
            self.close_video_recorder()
        self.render_rate = render
        if render:
            print("Seed:", seed)
        if video_save_dir:
            self.video_recorder = video_recorder(self, video_save_dir)
            self.render_rate = render*0.7
        return self.current_observation()

    def close_video_recorder(self):
        self.video_recorder.close()
        self.video_recorder = None

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
    def simulate(self, target, target_heading=None, action=None, put_dots=False):
        self.controller.set_gait_raw(raw_gait=action, target_heading=target_heading, target=target)
        steps_per_render = None
        if self.render_rate:
            steps_per_render = int(REAL_TIME_STEPS_PER_RENDER / self.render_rate)
            if put_dots:
                self.sdf_loader.put_dot(target, 'step_target', color=GREEN)
                self.sdf_loader.put_dot(self.controller.prev_target, 'prev_step_target', color=GREEN)
        while True:
            # We use %1 instead of %0 to avoid distracting jumps when resetting things like
            # ground platforms and dots.
            if steps_per_render and self.world.frame % steps_per_render == 1:
                self._render()
            obs, terminated, status_string = self.simulation_step()
            if obs is not None:
                if self.render_rate:
                    self._render()
                    self.log(status_string)
                return obs, terminated

    def get_x(self):
        c = self.consts()
        q = c.standardized_dofs(self.robot_skeleton.q)
        dq = c.standardized_dofs(self.robot_skeleton.dq)
        return q, dq

    def current_observation(self):
        obs = np.concatenate(self.get_x())
        obs = np.concatenate((obs, self.controller.state()))
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
        c = self.consts()
        if state is None:
            self.robot_skeleton.x += perturbation
            # Start with some forward momentum (Simbicon has some trouble otherwise)
            dq = np.zeros(ndofs)
            dq[0] += 0.4 + random * np.random.uniform(low=0.0, high=0.4)
            dq = self.robot_skeleton.dq + c.raw_dofs(dq)
            self.robot_skeleton.dq = dq
            self.controller.reset()
        else:
            # TODO should State objects contain *all* state, or just the reduced-dim
            # representation used by the FSM?
            perturbation[:ndofs] += c.raw_dofs(state.raw_state[:c.Q_DIM])
            perturbation[ndofs:] += c.raw_dofs(state.raw_state[c.Q_DIM:2*c.Q_DIM])
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

    def pause(self, sec=1.5):
        # Give the viewer some time to look around and maybe rotate the environment.
        # TODO have some kind of background thread do rendering, to avoid this hack
        # and make 3D environments easier to explore visually.
        FPS = 33
        n = int(FPS*sec)
        for i in range(n):
            self._render()
            time.sleep(1/FPS)

    def update_viewer(self, com):
        track_point = self.track_point or com
        #track_point = track_point + np.array([-2, 0, 0])
        #tb.trans[1] = -0.5
        self.viewer.scene.tb.trans[0] = -track_point[0]
        self.viewer.scene.tb.trans[2] = -self.zoom

    def gui(self):
        pydart.gui.viewer.launch(self.world)

    def load_robot(self, world):
        # Loading the robot is a separate step in DarwinEnv
        skel = world.skeletons[1]
        assert(skel.name != "doppelganger")
        # TODO I spent a whole day tracking down weird behavior that was ultimately due to an incorrect joint limit. Is there a way that I can visualize when these limits are being hit?
        for j in skel.joints:
            j.set_position_limit_enforced(True)
        return skel

    def load_world(self):
        c = self.consts()
        world = pydart.World(c.SIMULATION_RATE, c.skel_file)
        self.robot_skeleton = self.load_robot(world)
        self.doppelganger = None
        if len(world.skeletons) == 3:
            self.doppelganger = world.skeletons[2]
            assert(self.doppelganger.name == "doppelganger")
        return world

