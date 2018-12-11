import numpy as np

class State:
    def __init__(self, raw_state):
        self.raw_state = raw_state.copy()
        self.q_len = (len(self.raw_state) - 9)//2

    def starting_platforms(self):
        # These should be returned in "chronologial" order: most recent platform last.
        return [self.swing_platform(), self.stance_platform()]

    def pose(self):
        return self.raw_state[ 0:-9]

    def dq(self):
        return self.raw_state[self.q_len:2*self.q_len]

    def stance_heel_location(self):
        return self.raw_state[-9:-6]

    def stance_platform(self):
        return self.raw_state[-6:-3]

    def swing_platform(self):
        return self.raw_state[-3:  ]

    def crashed(self):
        return not np.isfinite(self.raw_state).all()
