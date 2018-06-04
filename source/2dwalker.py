import pydart2 as pydart
from IPython import embed
from enum import Enum
import numpy as np

EPISODE_TIME_LIMIT = 5.0 # seconds

class Controller:
    def __init__(self, skel, low_level_controller):
        self.skel = skel
        self.target = None
        self.inactive = False
        self.low_level_controller = low_level_controller
        brickdof = 3 # We're in 2D
        self.Kp = np.array([0.0] * brickdof + [400.0] * (self.skel.ndofs - brickdof))
        self.Kd = np.array([0.0] * brickdof + [40.0] * (self.skel.ndofs - brickdof))

    def compute(self):
        if self.inactive:
            return np.zeros_like(self.Kp)
        return -self.Kp * (self.skel.q - self.target) - self.Kd * self.skel.dq

class State(Enum):
    INIT = 0
    RIGHT_PLANT = 1
    RIGHT_SWING = 2
    LEFT_PLANT = 3
    LEFT_SWING = 4

class Episode:
    def __init__(self, world, low_level_controller):
        self.world = world
        self.target = 0.5
        self.score = 0.0
        walker = world.skeletons[1]
        self.r_foot = walker.bodynodes[5]
        self.l_foot = walker.bodynodes[8]

        controller = Controller(walker, low_level_controller)
        pos = walker.positions()
        pos[3] = 1
        # TODO: introduce some randomness to the starting conditions
        walker.set_positions(pos)
        # For now, just set a static PD controller target pose
        # TODO: Choose this target pose from some other policy
        controller.target = pos

        walker.set_controller(controller)
        self.state = State.INIT
        pydart.gui.viewer.launch(world)

    # We locate heeldown events for a given foot based on the first contact
    # for that foot after it has been in the air for at least one frame.
    def find_contact(self, bodynode):
        # TODO: It is possible for a body node to have multiple contacts.
        # Which one should be returned?
        for c in self.world.collision_result.contacts:
            if c.bodynode2 == bodynode:
                return c

    # On each heeldown event, we get points based on how close we were to
    # the target step location. Maximum 1 point per step.
    def calc_score(self, contact):
        # p[0] is the X position of the contact
        return np.exp(-(contact.p[0] - self.target)**2)

    def print_contact(self, contact):
        print(
            "{:.3f}, {:.3f}, {}: {} made contact at {:.3f} (target: {:.3f})".format(
            self.world.time(), self.score, self.state,
            contact.bodynode2, contact.p[0], self.target))

    # The episode terminates after one right footstep and one left footstep,
    # or if the time limit is reached.
    def run(self):
        while True:
            if self.world.time() > EPISODE_TIME_LIMIT:
                print("Time limit reached")
                return self.score

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
                self.print_contact(r_foot_contact)
                self.target += 0.5
            elif self.state == State.LEFT_PLANT and l_foot_contact is None:
                self.state = State.LEFT_SWING
            elif self.state == State.LEFT_SWING and l_foot_contact is not None:
                self.score += self.calc_score(l_foot_contact)
                self.print_contact(l_foot_contact)
                print("Finished episode")
                return self.score

            world.step()

if __name__ == '__main__':
    SKEL_ROOT = "/home/nathan/research/pydart2/examples/data/skel/"
    DARTENV_SKEL_ROOT = "/home/nathan/research/dart-env/gym/envs/dart/assets/"
    skel = DARTENV_SKEL_ROOT + "walker2d.skel"

    pydart.init()
    world = pydart.World(1.0 / 2000.0, skel)

    # Place visual markers at the target footstep locations
    for t in [0.5, 1.0]:
        dot = world.add_skeleton('./dot.sdf')
        q = dot.q
        q[3] = t
        dot.q = q

    ep = Episode(world, None)
    ep.run()

