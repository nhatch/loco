from IPython import embed
import numpy as np

BRICK_DOF = 3 # We're in 2D
KP_GAIN = 200.0
KD_GAIN = 15.0

# TODO implement Stable PD controllers?
class PDController:
    def __init__(self, skel, world):
        self.skel = skel
        self.world = world
        self.inactive = False
        self.reset()
        self.Kp = np.array([0.0] * BRICK_DOF + [KP_GAIN] * (self.skel.ndofs - BRICK_DOF))
        self.Kd = np.array([0.0] * BRICK_DOF + [KD_GAIN] * (self.skel.ndofs - BRICK_DOF))

    def set_actuated_pose(self, pose):
        self.target_q[BRICK_DOF:] = pose

    def compute(self):
        if self.inactive:
            return np.zeros_like(self.Kp)
        # TODO clamp control
        return -self.Kp * (self.skel.q - self.target_q) - self.Kd * self.skel.dq

    def state_complete(self, left, right):
        # Stub to conform to Simbicon interface
        pass

    def reset(self):
        self.target_q = self.skel.q
