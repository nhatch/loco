import numpy as np

# len(robot_skeleton.x)
X_LEN = 42

class State3D:
    def __init__(self, raw_state):
        self.raw_state = raw_state.copy()

    def starting_platforms(self):
        return [self.swing_platform(), self.stance_platform()]

    def pose(self):
        return self.raw_state[0:X_LEN]

    def controller_state(self):
        return self.raw_state[X_LEN:]

    def stance_contact_location(self):
        return self.raw_state[X_LEN:X_LEN+2]

    def stance_heel_location(self):
        return self.raw_state[X_LEN+2:X_LEN+4]

    def stance_platform(self):
        return self.raw_state[X_LEN+4:X_LEN+6]

    def swing_platform(self):
        return self.raw_state[X_LEN+6:X_LEN+8]

    def crashed(self):
        return not np.isfinite(self.raw_state).all()
