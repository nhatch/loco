import pydart2 as pydart
from IPython import embed
from enum import Enum
import numpy as np
import random_search

from simbicon import SIMBICON_ACTION_SIZE
from pd_control import PDController
from inverse_dynamics import flip_stance

# For rendering
from gym.envs.dart.static_window import *
from pydart2.gui.trackball import Trackball

SIMULATION_RATE = 1.0 / 2000.0 # seconds
EPISODE_TIME_LIMIT = 5.0 # seconds
LLC_QUERY_PERIOD = 1.0 / 30.0 # seconds
STEPS_PER_QUERY = int(LLC_QUERY_PERIOD / SIMULATION_RATE)

class TwoStepEnv:
    def __init__(self, controller_class, render_factor=1.0):
        world = load_world()
        self.world = world
        self.steps_per_render = int(STEPS_PER_QUERY / render_factor) + 1
        self.visual = (render_factor > 1.0)
        walker = world.skeletons[1]
        self.robot_skeleton = walker
        self.r_foot = walker.bodynodes[5]
        self.l_foot = walker.bodynodes[8]

        # Hacks to make this work with the gym.wrappers Monitor API
        self.metadata = {'render.modes': ['rgb_array', 'human']}
        self.reward_range = range(10)
        self.spec = None

        # We just want this to be something that has a "shape" method
        # TODO incorporate adding target step locations into the observations
        self.observation_space = np.zeros_like(self.current_observation())
        self.action_space = np.zeros(SIMBICON_ACTION_SIZE)

        self.controller = controller_class(walker, world)
        walker.set_controller(self.controller)
        self.reset_x = (walker.q.copy(), walker.dq.copy())
        #self.reset_x[0][3] = 1 # Experimenting with what the DOFs mean
        # For now, just set a static PD controller target pose

        title = None
        win = StaticGLUTWindow(self.world, title)
        # For some reason setting 'zoom' doesn't do anything
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)
        win.run()
        self.viewer = win

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.world.reset()
        self.controller.reset()
        self.place_footstep_targets([0.5])
        self.robot_skeleton.q = self.reset_x[0].copy() + np.random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.robot_skeleton.dq = self.reset_x[1].copy() + np.random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        return self.current_observation()

    # We locate heeldown events for a given foot based on the first contact
    # for that foot after it has been in the air for at least one frame.
    def find_contact(self, bodynode):
        # TODO: It is possible for a body node to have multiple contacts.
        # Which one should be returned?
        for c in self.world.collision_result.contacts:
            if c.bodynode2 == bodynode:
                return c

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

        l_contact = self.find_contact(self.l_foot)
        r_contact = self.find_contact(self.r_foot)
        state_complete, step_dist = self.controller.state_complete(l_contact, r_contact)
        if state_complete:
            self.controller.change_state()
            if step_dist:
                return step_dist

        self.world.step()

    def current_observation(self):
        # TODO remove the first element? (x location) -- it doesn't matter. The more important thing is perhaps x location relative to the stance foot?
        return self.robot_skeleton.x

    def log(self, string):
        print(string)

    # Run one footstep of simulation, returning the final state and the achieved step distance
    def simulate(self, start_state=None, action=None):
        if start_state is not None:
            self.robot_skeleton.x = start_state
        if action is not None:
            self.controller.set_gait_raw(action)
        while True:
            if self.visual and self.world.frame % self.steps_per_render == 0:
                self.render()
            step_dist = self.simulation_step()
            if type(step_dist) == str:
                self.log("ERROR: " + step_dist)
                return None, None
            if step_dist:
                end_state = self.robot_skeleton.x.copy()
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
                    if j % 2 == 0:
                        # This was a left foot swing, so flip it.
                        end_state = flip_stance(end_state)
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

