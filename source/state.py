import numpy as np

class State:
    def __init__(self, raw_state):
        self.raw_state = raw_state.copy()

    def starting_platforms(self):
        return [self.swing_platform(), self.stance_platform()]

    def pose(self):
        return self.raw_state[ 0:18]

    def controller_state(self):
        return self.raw_state[18:27]

    def stance_heel_location(self):
        return self.raw_state[18:21]

    def swing_platform(self):
        return self.raw_state[21:24]

    def stance_platform(self):
        return self.raw_state[24:27]

    def crashed(self):
        return not np.isfinite(self.raw_state).all()
