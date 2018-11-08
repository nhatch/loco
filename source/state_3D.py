import numpy as np

from consts_common3D import *
X_LEN = Q_DIM*2

class State3D:
    def __init__(self, raw_state):
        self.raw_state = raw_state.copy()

    def starting_platforms(self):
        return [self.swing_platform(), self.stance_platform()]

    def pose(self):
        return self.raw_state[0:X_LEN]

    def controller_state(self):
        return self.raw_state[X_LEN:]

    def stance_heel_location(self):
        return self.raw_state[X_LEN:X_LEN+3]

    def swing_platform(self):
        return self.raw_state[X_LEN+3:X_LEN+6]

    def stance_platform(self):
        return self.raw_state[X_LEN+6:X_LEN+9]

    def crashed(self):
        return not np.isfinite(self.raw_state).all()
