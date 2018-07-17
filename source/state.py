class State:
    def __init__(self, raw_state):
        self.raw_state = raw_state

    def starting_platforms(self):
        return [self.raw_state[24:26], self.raw_state[18:20]]
