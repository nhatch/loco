import numpy as np
from IPython import embed

from simbicon import Simbicon, UP, DOWN

class Simbicon3D(Simbicon):
    # TODO test this
    def standardize_stance(self, state):
        c = self.env.consts()
        state = super().standardize_stance(state)
        # Rotations are absolute, not relative, so we need to multiply some angles
        # by -1 to obtain a mirrored pose.
        D = c.Q_DIM
        m = np.ones(D)
        absolute_rotation_indices = [6,7,10,11,14,16,17,20]
        m[absolute_rotation_indices] = -1
        state[0:D] *= m
        state[D:2*D] *= m
        return state

    def maybe_start_down_phase(self, contacts, swing_heel):
        duration = self.time() - self.step_started
        if duration > 0.3:
            self.direction = DOWN


