from IPython import embed
import numpy as np

# TODO implement Stable PD controllers?
class PDController:
    def __init__(self, skel, env):
        self.skel = skel
        self.env = env
        self.inactive = False
        self.reset()

        c = self.env.consts()
        self.swing_idx = c.RIGHT_IDX # Stub to conform to Simbicon interface

        BRICK_DOF = c.BRICK_DOF
        self.Kp = np.concatenate([[0.0] * BRICK_DOF, c.KP_GAIN])
        self.Kd = np.concatenate([[0.0] * BRICK_DOF, c.KD_GAIN])
        self.control_bounds = c.CONTROL_BOUNDS

    def compute_PD(self, target_q):
        if self.inactive:
            return np.zeros_like(self.Kp)
        c = self.env.consts()
        BRICK_DOF = c.BRICK_DOF
        raw_target_q = c.raw_dofs(target_q)
        q, dq = self.skel.q, self.skel.dq
        control = -self.Kp * (q - raw_target_q) - self.Kd * dq
        for i in range(BRICK_DOF, len(control)):
            if control[i] > self.control_bounds[1][i-BRICK_DOF]:
                control[i] = self.control_bounds[1][i-BRICK_DOF]
            if control[i] < self.control_bounds[0][i-BRICK_DOF]:
                control[i] = self.control_bounds[0][i-BRICK_DOF]
        return np.array(control)

    def compute(self):
        return self.compute_PD(self.target_q)

    def reset(self, state=None):
        self.target_q = self.env.consts().standardized_dofs(np.array(self.skel.q))

    def state(self):
        return []
