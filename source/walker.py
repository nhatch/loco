import pydart2 as pydart
from IPython import embed
from enum import Enum
import numpy as np
import random_search

from simbicon import SIMBICON_ACTION_SIZE
from pd_control import PDController

# For rendering
from gym.envs.dart.static_window import *
from pydart2.gui.trackball import Trackball

SIMULATION_RATE = 1.0 / 2000.0 # seconds
EPISODE_TIME_LIMIT = 5.0 # seconds
REAL_TIME_STEPS_PER_RENDER = 25 # Number of simulation steps to run per frame so it looks like real time. Just a rough estimate.

class TwoStepEnv:
    def __init__(self, controller_class):
        world = load_world()
        self.world = world
        walker = world.skeletons[1]
        self.robot_skeleton = walker
        self.r_foot = walker.bodynodes[5]
        self.l_foot = walker.bodynodes[8]

        self.controller = controller_class(walker, self)
        walker.set_controller(self.controller)
        self.reset_x = (walker.q.copy(), walker.dq.copy())
        #self.reset_x[0][3] = 1 # Experimenting with what the DOFs mean
        # For now, just set a static PD controller target pose

        # We just want this to be something that has a "shape" method
        # TODO incorporate adding target step locations into the observations
        self.observation_space = np.zeros_like(self.current_observation())
        self.action_space = np.zeros(SIMBICON_ACTION_SIZE)

        # Hacks to make this work with the gym.wrappers Monitor API
        self.metadata = {'render.modes': ['rgb_array', 'human']}
        self.reward_range = range(10)
        self.spec = None

        title = None
        win = StaticGLUTWindow(self.world, title)
        # For some reason setting 'zoom' doesn't do anything
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)
        win.run()
        self.viewer = win

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, state=None):
        self.world.reset()
        self.controller.reset()
        self.place_footstep_targets([0.5])
        self.set_state(state)
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

    # The episode terminates after one right footstep and one left footstep,
    # or if the time limit is reached.
    def simulation_step(self):
        if self.world.time() > EPISODE_TIME_LIMIT:
            return "Time limit reached"
        obs = self.current_observation()
        if not np.isfinite(obs).all():
            return "Numerical explosion"
        if self.crashed():
            return "Crashed"

        l_contact = self.find_contacts(self.l_foot)
        r_contact = self.find_contacts(self.r_foot)
        state_complete, step_dist = self.controller.state_complete(l_contact, r_contact)
        if state_complete:
            self.controller.change_state()
            if step_dist:
                return step_dist

        self.world.step()

    def current_observation(self):
        obs = self.robot_skeleton.x.copy()
        # Absolute x location doesn't matter -- just location relative to the stance contact
        obs[0] -= self.controller.contact_x
        return self.standardize_stance(obs)

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

    def set_state(self, state):
        if state is None:
            self.robot_skeleton.q = self.reset_x[0].copy() + np.random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            dq = self.reset_x[1].copy() + np.random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
            # Start with some forward momentum (Simbicon has some trouble otherwise)
            dq[0] += np.random.uniform(low=0.25, high=1.5)
            self.robot_skeleton.dq = dq
        else:
            self.robot_skeleton.x = state
            self.controller.contact_x = 0.0 # Relative to state[0] (cf current_observation)

    # Run one footstep of simulation, returning the final state and the achieved step distance
    def simulate(self, action=None, render=False):
        steps_per_render = None
        if render:
            steps_per_render = int(REAL_TIME_STEPS_PER_RENDER / render)
        if action is not None:
            self.controller.set_gait_raw(action)
        while True:
            if steps_per_render and self.world.frame % steps_per_render == 0:
                self.render()
            step_dist = self.simulation_step()
            if type(step_dist) == str:
                self.log("ERROR: " + step_dist)
                return None, None
            if step_dist:
                end_state = self.current_observation()
                return end_state, step_dist

    def step(self, action):
        end_state, step_dist = self.simulate(action=action)
        if step_dist is not None:
            # Reward is step distance for now . . . this isn't meant to be used as RL though.
            return self.current_observation(), step_dist, False, {}
        return self.current_observation(), 0.0, True, {}

    def collect_starting_states(self, size=8, n_resets=4):
        self.log("Collecting initial starting states")
        start_states = []
        self.controller.set_gait_raw(np.zeros(self.action_space.shape[0]))
        for i in range(n_resets):
            self.log("Starting trajectory {}".format(i))
            self.reset()
            # TODO should we include this first state? It will be very different from the rest.
            #start_states.append(self.robot_skeleton.x)
            for j in range(size):
                end_state, _ = self.simulate()
                if end_state is not None:
                    start_states.append(end_state)
        return start_states

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

    def put_dot(self, x, y, idx=0):
        dot = self.world.skeletons[2+idx]
        q = dot.q
        q[3] = x
        q[4] = y
        dot.q = q

    def place_footstep_targets(self, targets):
        # Place visual markers at the target footstep locations
        # TODO: allow placing variable numbers of dots in the environment
        for i, t in enumerate(targets):
            self.put_dot(t, 0, i)

def load_world():
    SKEL_ROOT = "/home/nathan/research/pydart2/examples/data/skel/"
    DARTENV_SKEL_ROOT = "/home/nathan/research/dart-env/gym/envs/dart/assets/"
    skel = DARTENV_SKEL_ROOT + "walker2d.skel"

    pydart.init()
    world = pydart.World(SIMULATION_RATE, skel)
    # These will mark footstep target locations
    # TODO: allow placing variable numbers of dots in the environment
    dot = world.add_skeleton('./dot.sdf')
    #dot = world.add_skeleton('./dot.sdf') # A second dot, not used yet

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

