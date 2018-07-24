from IPython import embed
import numpy as np

KP_GAIN = 200.0
KD_GAIN = 15.0

control_bounds_2D = 1.5 * np.array([100, 100, 20, 100, 100, 20])
control_bounds_3D = 1.5 * np.array([100]*15)

# TODO implement Stable PD controllers?
class PDController:
    def __init__(self, skel, env):
        self.skel = skel
        self.env = env
        self.inactive = False
        self.reset()

        brick_dof = self.env.brick_dof
        self.Kp = np.array([0.0] * brick_dof + [KP_GAIN] * (self.skel.ndofs - brick_dof))
        self.Kd = np.array([0.0] * brick_dof + [KD_GAIN] * (self.skel.ndofs - brick_dof))
        self.control_bounds = control_bounds_2D if brick_dof == 3 else control_bounds_3D

    def compute(self):
        brick_dof = self.env.brick_dof
        if self.inactive:
            return np.zeros_like(self.Kp)
        control = -self.Kp * (self.skel.q - self.target_q) - self.Kd * self.skel.dq
        for i in range(brick_dof, self.skel.ndofs):
            if control[i] > self.control_bounds[i-brick_dof]:
                control[i] = self.control_bounds[i-brick_dof]
            if control[i] < -self.control_bounds[i-brick_dof]:
                control[i] = -self.control_bounds[i-brick_dof]
        return control

    def step_complete(self, contacts, swing_heel):
        # Stub to conform to Simbicon interface
        return False

    def reset(self):
        self.target_q = self.skel.q

    def state(self):
        return []

    def standardize_stance(self, state):
        return state
