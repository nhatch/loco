import os

# SDF color format is space-separated RGBA
# Here are some suggested settings
RED = "0.8 0 0 1"
GREEN = "0.2 0.7 0.2 1"
BLUE = "0.4 0.4 1 1"

class SDFLoader:
    def __init__(self, world):
        self.world = world
        self.num_dots = 0
        self.num_grounds = 0

    def put_dot(self, x, y, color=RED):
        # Change the skeleton name so that the console output is not cluttered
        # with warnings about duplicate names.
        os.system("sed -e 's/__NAME__/dot_" + str(self.num_dots) + "/' dot.sdf > _dot.sdf")
        os.system("sed -e 's/__COLOR__/" + color + "/' _dot.sdf > __dot.sdf")
        self.num_dots += 1

        dot = self.world.add_skeleton('./__dot.sdf')
        q = dot.q
        q[3] = x
        q[4] = y
        dot.q = q

    # Length is in meters.
    def put_ground(self, x, length):
        # Change the skeleton name so that the console output is not cluttered
        # with warnings about duplicate names.
        os.system("sed -e 's/__NAME__/ground_" + str(self.num_grounds)
                    + "/' ground.sdf > _ground.sdf")
        os.system("sed -e 's/__LEN__/" + str(length) + "/' _ground.sdf > __ground.sdf")
        self.num_grounds += 1

        ground = self.world.add_skeleton('./__ground.sdf')
        q = ground.q
        # The x coordinate q[3] gives the *center* of the block.
        q[3] = x + 0.5*length
        ground.q = q

if __name__ == "__main__":
    from walker import TwoStepEnv
    from simbicon import Simbicon
    env = TwoStepEnv(Simbicon)
    env.reset()
    env.gui()
