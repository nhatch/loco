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
KP_GAIN = 200.0
KD_GAIN = 15.0

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
        self.target = 0

        # We just want this to be something that has a "shape" method
        # TODO incorporate adding target step locations into the observations
        self.observation_space = np.zeros_like(self.current_observation())
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

    def inv_kine(self, targ_down, targ_forward):
        L_LEG = 0.45
        L_SHIN = 0.5

        r = np.sqrt(targ_down**2 + targ_forward**2)
        cos_knee = (r**2 - L_LEG**2 - L_SHIN**2) / (2 * L_LEG * L_SHIN)
        tan_theta3 = targ_forward / targ_down
        cos_theta4 = (r**2 + L_LEG**2 - L_SHIN**2) / (2 * L_LEG * r)

        # Handle out-of-reach targets by clamping cosines to 1
        if cos_knee > 1:
            cos_knee = 1
        if cos_theta4 > 1:
            cos_theta4 = 1

        # We want knee angle to be < 0
        knee = - np.arccos(cos_knee)
        # The geometry makes more sense if theta4 is > 0
        theta4 = np.arccos(cos_theta4)
        # Since knee < 0, we add theta3 and theta4 to get hip angle
        hip = np.arctan(tan_theta3) + theta4
        return hip, knee

    def inv_kine_pose(self, targ_down, targ_forward, is_right_foot=True):
        hip, knee = self.inv_kine(targ_down, targ_forward)
        q = self.agent.q.copy()
        if is_right_foot:
            q[3] = hip
            q[4] = knee
        else: # left foot
            q[6] = hip
            q[7] = knee
        return q

    def test_inv_kine(self):
        # Generate a random starting pose and target (marked with a dot),
        # then edit the right hip and knee joints to hit the target.
        center = np.array([0, 0.3, 0])
        pose = center + np.random.uniform(low=-0.2, high = 0.2, size = 3)
        q = self.agent.q
        q[:3] = pose
        self.agent.q = q
        target = center + np.random.uniform(low=-0.5, high=0.5, size=3)
        target[1] += 0.5
        self.put_dot(target[0], target[1])
        print("TARGET:", target[0], target[1])
        down, forward = self.transform_frame(target[0], target[1], verbose=True)
        q = self.inv_kine_pose(down, forward)
        self.agent.q = q
        self.render()

    def transform_frame(self, x, y, verbose=False):
        # Takes the absolute coordinates (x,y) and transforms them into (down, forward)
        # relative to the pelvis joint's absolute location and rotation.
        pelvis_com = self.agent.bodynodes[2].com()
        theta = self.agent.q[2]
        L_PELVIS = 0.4
        self.put_dot(pelvis_com[0], pelvis_com[1], 1)
        if verbose:
            print("PELVIS COM:", pelvis_com[0], pelvis_com[1])
        # Put pelvis COM at origin
        x = x - pelvis_com[0]
        y = y - pelvis_com[1]
        # Transform to polar coordinates
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan(y/x)
        if phi*y < 0:
            phi = phi - np.pi
        if verbose:
            print("ANGLES:", phi, theta)
        # Rotate
        phi = phi - theta
        # Transform back to Euclidean coordinates
        down = -r * np.sin(phi)
        forward = r * np.cos(phi)
        # Use bottom of pelvis instead of COM
        down = down - L_PELVIS / 2
        if verbose:
            print("RELATIVE TO JOINT:", down, forward)
        return down, forward

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
        d = np.abs(contact.p[0] - self.target)
        #placement_score = np.exp(-10 * d)
        #placement_score = np.exp(-d**2)
        placement_score = 1 - d
        # TODO: penalize large actuation and/or hitting joint limits?
        return placement_score

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
        return np.concatenate((self.agent.x, [self.target]))

    def log(self, string):
        print(string)

    def step(self, action):
        self.controller.target_q[BRICK_DOF:] = action
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
        #self.viewer.scene.tb.trans[0] = -self.agent.com()[0]*1
        self.viewer.scene.tb.trans[2] = -5 # adjust zoom
        self.viewer.runSingleStep()

    def put_dot(self, x, y, idx=0):
        dot = self.world.skeletons[2+idx]
        q = dot.q
        q[3] = x
        q[4] = y
        dot.q = q

    def place_footstep_targets(self, targets):
        # Place visual markers at the target footstep locations
        for i, t in enumerate(targets):
            self.put_dot(t, 0, i)

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
    embed()

