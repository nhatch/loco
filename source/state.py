import numpy as np

class State:
    def __init__(self, raw_state):
        self.raw_state = raw_state.copy()

    def starting_platforms(self):
        return [self.swing_platform(), self.stance_platform()]

    def pose(self):
        return self.raw_state[ 0:18]

    def controller_state(self):
        return self.raw_state[18:26]

    def stance_contact_location(self):
        return self.raw_state[18:20]

    def stance_heel_location(self):
        return self.raw_state[20:22]

    def stance_platform(self):
        return self.raw_state[22:24]

    def swing_platform(self):
        return self.raw_state[24:26]

    def crashed(self):
        return not np.isfinite(self.raw_state).all()
