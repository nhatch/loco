import pydart2 as pydart
from IPython import embed
from enum import Enum
import numpy as np
import os
import random_search

from simbicon import SIMBICON_ACTION_SIZE
from pd_control import PDController
from video_recorder import video_recorder

# For rendering
from gym.envs.dart.static_window import *
from pydart2.gui.trackball import Trackball

SIMULATION_RATE = 1.0 / 2000.0 # seconds
EPISODE_TIME_LIMIT = 80.0 # seconds
REAL_TIME_STEPS_PER_RENDER = 25 # Number of simulation steps to run per frame so it looks like real time. Just a rough estimate.

FRICTION_COEFF = 1.0

class StepResult(Enum):
    ERROR = -1
    IN_PROGRESS = 0
    COMPLETE = 1

class TwoStepEnv:
    def __init__(self, controller_class):
        world = load_world()
        self.world = world
        self.num_dots = 0
        walker = world.skeletons[1]
        self.robot_skeleton = walker
        self.r_foot = walker.bodynodes[5]
        self.l_foot = walker.bodynodes[8]
        # TODO: somehow, despite this, the feet are still slipping.
        # It also makes the walker trip unless I increase the PD gains.
        walker.bodynodes[5].set_friction_coeff(FRICTION_COEFF)
        walker.bodynodes[8].set_friction_coeff(FRICTION_COEFF)
        world.skeletons[0].bodynodes[0].set_friction_coeff(FRICTION_COEFF)
        for j in walker.joints:
            j.set_position_limit_enforced()

        self.controller = controller_class(walker, self)
        walker.set_controller(self.controller)

        # We just want this to be something that has a "shape" method
        # TODO incorporate adding target step locations into the observations
        self.observation_space = np.zeros_like(self.current_observation())
        self.action_space = np.zeros(SIMBICON_ACTION_SIZE)

        self.video_recorder = None
        # Hacks to make this work with the gym.wrappers Monitor API
        self.metadata = {'render.modes': ['rgb_array', 'human']}
        #self.reward_range = range(10)
        #self.spec = None

        title = None
        win = StaticGLUTWindow(self.world, title)
        # For some reason setting 'zoom' doesn't do anything
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)
        win.run()
        self.viewer = win

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None, record_video=False, random=1.0):
        self.world.reset()
        # TODO remove extra dot skeletons?
        self.controller.reset()
        self.set_state(state, random)
        if self.video_recorder:
            self.video_recorder.close()
        if record_video:
            self.video_recorder = video_recorder(self)
        else:
            self.video_recorder = None
        return self.current_observation()

    def find_contacts(self, bodynode):
        return [c for c in self.world.collision_result.contacts if c.bodynode2 == bodynode]

    def crashed(self):
        allowed_contact = [
                'ground',
                'h_shin',
                'h_foot',
                'h_shin_left',
                'h_foot_left']
        for b in self.world.collision_result.contacted_bodies:
            if not (b.name in allowed_contact):
                return True

    # Executes one world step.
    # Returns (status code, step distance, human-readable message) tuple.
    def simulation_step(self):
        if self.world.time() > EPISODE_TIME_LIMIT:
            return StepResult.ERROR, None, "Time limit reached"
        obs = self.current_observation()
        if not np.isfinite(obs).all():
            return StepResult.ERROR, None, "Numerical explosion"
        if self.crashed():
            return StepResult.ERROR, None, "Crashed"

        l_contact = self.find_contacts(self.l_foot)
        r_contact = self.find_contacts(self.r_foot)
        state_complete, step_dist = self.controller.state_complete(l_contact, r_contact)
        if state_complete:
            status_string = self.controller.change_state()
            if step_dist:
                return StepResult.COMPLETE, step_dist, status_string

        self.world.step()
        return StepResult.IN_PROGRESS, None, None

    def current_observation(self):
        obs = self.robot_skeleton.x.copy()
        obs = self.standardize_stance(obs)
        obs[0] = 0.0 # Exact x location doesn't affect dynamics; let's ignore this
        base_x = self.robot_skeleton.x[0]
        stance = np.array([self.controller.contact_x, self.controller.stance_heel]) - base_x
        return np.concatenate((obs, stance))

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
        else:
            self.robot_skeleton.x = state[:18]
            self.controller.contact_x = state[18]
            self.controller.stance_heel = state[19]

    # Run one footstep of simulation, returning the final state and the achieved step distance
    def simulate(self, action=None, render=False, target_x=None):
        if target_x is not None:
            self.put_dot(target_x, 0)
        steps_per_render = None
        if render:
            steps_per_render = int(REAL_TIME_STEPS_PER_RENDER / render)
        if action is not None:
            self.controller.set_gait_raw(action)
        if target_x is not None:
            self.controller.target_x = target_x
        while True:
            if steps_per_render and self.world.frame % steps_per_render == 0:
                self._render()
            status_code, step_dist, status_string = self.simulation_step()
            if status_code == StepResult.ERROR:
                self.log("ERROR: " + status_string)
                return None, None
            if status_code == StepResult.COMPLETE:
                end_state = self.current_observation()
                if render:
                    self.log(status_string)
                return end_state, step_dist

    def collect_starting_states(self, size=8, n_resets=16):
        self.log("Collecting initial starting states")
        start_states = []
        self.controller.set_gait_raw(np.zeros(self.action_space.shape[0]))
        for i in range(n_resets):
            self.log("Starting trajectory {}".format(i))
            self.reset()
            # TODO should we include this first state? It will be very different from the rest.
            #start_states.append(self.robot_skeleton.x)
            for j in range(size):
                end_state, _ = self.simulate(render=1.0)
                if end_state is not None:
                    start_states.append(end_state)
        return start_states

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

    def put_dot(self, x, y):
        # Change the skeleton name so that the console output is not cluttered
        # with warnings about duplicate names.
        os.system("sed -e 's/dot/dot_" + str(self.num_dots) + "/' dot.sdf > _dot.sdf")
        self.num_dots += 1

        dot = self.world.add_skeleton('./_dot.sdf')
        q = dot.q
        q[3] = x
        q[4] = y
        dot.q = q

def load_world():
    skel = "walker2d.skel"
    pydart.init(verbose=False)
    world = pydart.World(SIMULATION_RATE, skel)
    return world

if __name__ == '__main__':
    # TODO this code doesn't work anymore since the action interface changed
    # I should probably write two separate environments (one for raw torques, one for Simbicon parameters)
    from inverse_kinematics import InverseKinematics
    env = TwoStepEnv(PDController)
    ik = InverseKinematics(env, 0.5)
    w = random_search.Whitener(env, False)
    for t in [0.5, 1.0, 0.2]:
        ik.target = t
        w.run_trajectory(ik.controller, 0, True, False)
    w.close()

