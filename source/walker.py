import pydart2 as pydart
from IPython import embed
from enum import Enum
import numpy as np
import random_search

# For rendering
from gym.envs.dart.static_window import *
from pydart2.gui.trackball import Trackball

SIMULATION_RATE = 1.0 / 2000.0 # seconds
EPISODE_TIME_LIMIT = 5.0 # seconds
LLC_QUERY_PERIOD = 1.0 / 30.0 # seconds
STEPS_PER_QUERY = int(LLC_QUERY_PERIOD / SIMULATION_RATE)
STEPS_PER_RENDER = STEPS_PER_QUERY // 4
BRICK_DOF = 3 # We're in 2D
KP_GAIN = 100.0
KD_GAIN = 10.0

class Controller:
    def __init__(self, skel):
        self.skel = skel
        self.target_q = None
        self.inactive = False
        self.Kp = np.array([0.0] * BRICK_DOF + [KP_GAIN] * (self.skel.ndofs - BRICK_DOF))
        self.Kd = np.array([0.0] * BRICK_DOF + [KD_GAIN] * (self.skel.ndofs - BRICK_DOF))

    def compute(self):
        if self.inactive:
            return np.zeros_like(self.Kp)
        # TODO clamp control
        return -self.Kp * (self.skel.q - self.target_q) - self.Kd * self.skel.dq

class State(Enum):
    INIT = 0
    RIGHT_PLANT = 1
    RIGHT_SWING = 2
    LEFT_PLANT = 3
    LEFT_SWING = 4

class TwoStepEnv:
    def __init__(self, world):
        self.world = world
        walker = world.skeletons[1]
        self.agent = walker
        self.r_foot = walker.bodynodes[5]
        self.l_foot = walker.bodynodes[8]

        # We just want this to be something that has a "shape" method
        # TODO incorporate adding target step locations into the observations
        self.observation_space = np.zeros(18)
        self.action_space = np.zeros(6)

        self.controller = Controller(walker)
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
        self.place_footstep_targets([0.5, 1.0])
        self.agent.q = self.reset_x[0].copy() + np.random.uniform(low=-.005, high=.005, size=self.agent.ndofs)
        self.agent.dq = self.reset_x[1].copy() + np.random.uniform(low=-.005, high=.005, size=self.agent.ndofs)
        # TODO make step targets random as well
        # (this will require providing the target as an observation)
        self.target = 0.5
        self.score = 0.0
        self.state = State.INIT
        self.controller.target_q = self.reset_x[0].copy()
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

    # On each heeldown event, we get points based on how close we were to
    # the target step location. Maximum 1 point per step.
    def calc_score(self, contact):
        # p[0] is the X position of the contact
        # TODO: penalize large actuation and/or hitting joint limits?
        return np.exp(-10 * np.abs(contact.p[0] - self.target))

    def log_contact(self, contact):
        self.log(
            "{:.3f}, {:.3f}, {}: {} made contact at {:.3f} (target: {:.3f})".format(
            self.world.time(), self.score, self.state,
            contact.bodynode2, contact.p[0], self.target))

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

        l_foot_contact = self.find_contact(self.l_foot)
        r_foot_contact = self.find_contact(self.r_foot)

        if self.state == State.INIT and l_foot_contact is not None:
            if r_foot_contact is not None:
                self.state = State.RIGHT_PLANT
            else:
                self.state = State.RIGHT_SWING
        elif self.state == State.RIGHT_PLANT and r_foot_contact is None:
            self.state = State.RIGHT_SWING
        elif self.state == State.RIGHT_SWING and r_foot_contact is not None:
            if l_foot_contact is not None:
                self.state = State.LEFT_PLANT
            else:
                self.state = State.LEFT_SWING
            self.score += self.calc_score(r_foot_contact)
            self.log_contact(r_foot_contact)
            self.target += 0.5
        elif self.state == State.LEFT_PLANT and l_foot_contact is None:
            self.state = State.LEFT_SWING
        elif self.state == State.LEFT_SWING and l_foot_contact is not None:
            self.score += self.calc_score(l_foot_contact)
            self.log_contact(l_foot_contact)
            return "Finished episode"

        world.step()

    def current_observation(self):
        return self.agent.x

    def log(self, string):
        print(string)

    def step(self, action):
        self.controller.target_q[BRICK_DOF:] = action
        done = False
        for i in range(STEPS_PER_QUERY):
            #if i % STEPS_PER_RENDER == 0:
            #    self.render() # For debugging -- if weird things happen between LLC actions
            result = self.simulation_step()
            if result:
                self.log("{}: {}".format(result, self.score))
                return self.current_observation(), self.score, True, None
        return self.current_observation(), 0, False, None

    def render(self):
        #pydart.gui.viewer.launch(world)
        self.viewer.scene.tb.trans[0] = -self.agent.com()[0]*1
        self.viewer.scene.tb.trans[2] = -5 # adjust zoom
        self.viewer.runSingleStep()

    def place_footstep_targets(self, targets):
        # Place visual markers at the target footstep locations
        for i, t in enumerate(targets):
            dot = self.world.skeletons[2 + i]
            q = dot.q
            q[3] = t
            dot.q = q

def load_world():
    SKEL_ROOT = "/home/nathan/research/pydart2/examples/data/skel/"
    DARTENV_SKEL_ROOT = "/home/nathan/research/dart-env/gym/envs/dart/assets/"
    skel = DARTENV_SKEL_ROOT + "walker2d.skel"

    pydart.init()
    world = pydart.World(SIMULATION_RATE, skel)
    # These will mark footstep target locations
    dot = world.add_skeleton('./dot.sdf')
    dot = world.add_skeleton('./dot.sdf')

    return world

if __name__ == '__main__':
    world = load_world()
    env = TwoStepEnv(world)
    rs = random_search.RandomSearch(env, 8, step_size=0.3, eps=0.3)
    rs.random_search(10)
    embed()

