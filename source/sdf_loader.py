import os
from IPython import embed

# SDF color format is space-separated RGBA
# Here are some suggested settings
RED = "0.8 0 0 1"
GREEN = "0.2 0.7 0.2 1"
BLUE = "0.4 0.4 1 1"

class SDFLoader:
    def __init__(self):
        self.ground_length = 0.1
        self.ground_offset = 0.02

    def reset(self, world):
        self.world = world
        self.num_dots = 0
        self.grounds = []

    def put_dot(self, target, color=RED):
        # Change the skeleton name so that the console output is not cluttered
        # with warnings about duplicate names.
        os.system("sed -e 's/__NAME__/dot_" + str(self.num_dots) + "/' skel/dot.sdf > skel/_dot.sdf")
        os.system("sed -e 's/__COLOR__/" + color + "/' skel/_dot.sdf > skel/__dot.sdf")
        self.num_dots += 1

        dot = self.world.add_skeleton('./skel/__dot.sdf')
        q = dot.q
        q[3:5] = target
        dot.q = q

    # Length is in meters.
    def put_ground(self, x, y, length, index):
        num_grounds = len(self.grounds)
        if num_grounds <= index:
            # Change the skeleton name so that the console output is not cluttered
            # with warnings about duplicate names.
            os.system("sed -e 's/__NAME__/ground_" + str(num_grounds)
                        + "/' skel/ground.sdf > skel/_ground.sdf")
            os.system("sed -e 's/__LEN__/" + str(length) + "/' skel/_ground.sdf > skel/__ground.sdf")

            self.grounds.append(self.world.add_skeleton('./skel/__ground.sdf'))

        ground = self.grounds[index]
        q = ground.q
        # The x coordinate q[3] gives the *center* of the block.
        q[3] = x + 0.5*length
        q[4] = y
        ground.q = q

    def put_grounds(self, targets, runway_length=None):
        for i in range(len(targets)):
            x, y = targets[i]
            length = self.ground_length
            if i == 0 and runway_length is not None:
                length = runway_length
            self.put_ground(x - self.ground_offset, y, length, i)

if __name__ == "__main__":
    from walker import TwoStepEnv
    from simbicon import Simbicon
    env = TwoStepEnv(Simbicon)
    env.reset()
    env.gui()
