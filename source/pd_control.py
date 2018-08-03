from IPython import embed
import numpy as np

control_bounds_2D = 1.5 * np.array([100, 100, 20, 100, 100, 20])
control_bounds_3D = 1.5 * np.array([100]*15)

# TODO implement Stable PD controllers?
class PDController:
    def __init__(self, skel, env):
        self.skel = skel
        self.env = env
        self.inactive = False
        self.reset()

        c = self.env.consts()
        BRICK_DOF = c.BRICK_DOF
        self.Kp = np.array([0.0] * BRICK_DOF + [c.KP_GAIN] * (self.skel.ndofs - BRICK_DOF))
        self.Kd = np.array([0.0] * BRICK_DOF + [c.KD_GAIN] * (self.skel.ndofs - BRICK_DOF))
        #if c.BRICK_DOF == 6:
        #    # I think these "muscles" need to be stronger.
        #    self.Kp[c.RIGHT_IDX + c.HIP_OFFSET_LAT] *= 4
        #    self.Kp[c.LEFT_IDX + c.HIP_OFFSET_LAT] *= 4
        self.control_bounds = control_bounds_2D if BRICK_DOF == 3 else control_bounds_3D

    def compute(self):
        BRICK_DOF = self.env.consts().BRICK_DOF
        if self.inactive:
            return np.zeros_like(self.Kp)
        control = -self.Kp * (self.skel.q - self.target_q) - self.Kd * self.skel.dq
        for i in range(BRICK_DOF, self.skel.ndofs):
            if control[i] > self.control_bounds[i-BRICK_DOF]:
                control[i] = self.control_bounds[i-BRICK_DOF]
            if control[i] < -self.control_bounds[i-BRICK_DOF]:
                control[i] = -self.control_bounds[i-BRICK_DOF]
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
