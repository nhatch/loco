from IPython import embed
import numpy as np

# TODO there is some foot slip happening. Is this because these gains are too high?
BRICK_DOF = 3 # We're in 2D
KP_GAIN = 200.0
KD_GAIN = 15.0

# TODO implement Stable PD controllers?
class PDController:
    def __init__(self, skel, env):
        self.skel = skel
        self.env = env
        self.inactive = False
        self.reset()
        self.Kp = np.array([0.0] * BRICK_DOF + [KP_GAIN] * (self.skel.ndofs - BRICK_DOF))
        self.Kd = np.array([0.0] * BRICK_DOF + [KD_GAIN] * (self.skel.ndofs - BRICK_DOF))
        self.control_bounds = 1.5 * np.array([100, 100, 20, 100, 100, 20])

    def set_actuated_pose(self, pose):
        self.target_q[BRICK_DOF:] = pose

    def compute(self):
        if self.inactive:
            return np.zeros_like(self.Kp)
        control = -self.Kp * (self.skel.q - self.target_q) - self.Kd * self.skel.dq
        for i in range(BRICK_DOF, self.skel.ndofs):
            if control[i] > self.control_bounds[i-BRICK_DOF]:
                control[i] = self.control_bounds[i-BRICK_DOF]
            if control[i] < -self.control_bounds[i-BRICK_DOF]:
                control[i] = -self.control_bounds[i-BRICK_DOF]
        return control

    def state_complete(self, left, right):
        # Stub to conform to Simbicon interface
        pass

    def reset(self):
        self.target_q = self.skel.q
