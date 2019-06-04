import os
import numpy as np
from IPython import embed
import sys

# Otherwise if running multiple experiments simultaneously,
# they might overwrite each other's SDF files.
UQ_ID = sys.argv[1] if len(sys.argv) > 1 else ''

# SDF color format is space-separated RGBA
# Here are some suggested settings
RED = "0.8 0 0 1"
GREEN = "0.2 0.7 0.2 1"
BLUE = "0.4 0.4 1 1"

class SDFLoader:
    def __init__(self, consts):
        self.ground_length = 10.0
        self.ground_offset = consts.GROUND_OFFSET
        self.ground_thickness = 0.05
        self.ground_width = consts.DEFAULT_GROUND_WIDTH
        self.consts = consts

    def reset(self, world):
        self.world = world
        self.dots = {}
        self.grounds = []

    def put_dot(self, target, name, color=RED):
        # Change the skeleton name so that the console output is not cluttered
        # with warnings about duplicate names.
        dot = self.dots.get(name)
        radius = self.consts.DOT_RADIUS
        if dot is None:
            os.system("sed -e 's/__NAME__/dot_{}/' skel/dot.sdf > skel/_dot_{}.sdf".format(
                name, UQ_ID))
            os.system("sed -e 's/__COLOR__/{}/' skel/_dot_{}.sdf > skel/__dot_{}.sdf".format(
                color, UQ_ID, UQ_ID))
            os.system("sed -e 's/__RADIUS__/{}/' skel/__dot_{}.sdf > skel/___dot_{}.sdf".format(
                radius, UQ_ID, UQ_ID))
            dot = self.world.add_skeleton('./skel/___dot_{}.sdf'.format(UQ_ID))
            self.dots[name] = dot
            dot.set_root_joint_to_trans_and_euler()
            os.system("rm skel/_*_{}.sdf".format(UQ_ID))
        dot.q = self.consts.convert_root(np.concatenate([target, [0,0,0]]))

    # Length is in meters.
    def _put_ground(self, target, length, width, index):
        num_grounds = len(self.grounds)
        if num_grounds <= index:
            # Change the skeleton name so that the console output is not cluttered
            # with warnings about duplicate names.
            os.system("sed -e 's/__NAME__/ground_{}/' skel/ground.sdf > skel/_ground_{}.sdf".format(
                str(num_grounds), UQ_ID))
            os.system("sed -e 's/__LEN__/{}/' skel/_ground_{}.sdf > skel/__ground_{}.sdf".format(
                str(length), UQ_ID, UQ_ID))
            os.system("sed -e 's/__WIDTH__/{}/' skel/__ground_{}.sdf > skel/___ground_{}.sdf".format(
                str(width), UQ_ID, UQ_ID))

            ground = self.world.add_skeleton('./skel/___ground_{}.sdf'.format(UQ_ID))
            ground.set_root_joint_to_trans_and_euler()
            self.grounds.append(ground)

        ground = self.grounds[index]
        q = np.concatenate([target, [0,0,0]])
        # The x coordinate q[0] gives the *center* of the block.
        q[0] += 0.5*length - self.ground_offset
        q[1] -= 0.5*self.ground_thickness
        ground.q = self.consts.convert_root(q)

    def put_grounds(self, targets):
        for i in range(len(targets)):
            length = self.ground_length
            width = self.ground_width
            self._put_ground(targets[i], length, width, i)

if __name__ == "__main__":
    from stepping_stones_env import SteppingStonesEnv
    from simbicon import Simbicon
    env = SteppingStonesEnv()
    env.reset()
    env.gui()
