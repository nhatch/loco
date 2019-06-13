import numpy as np
import utils
from IPython import embed

class State:
    def __init__(self, raw_state, swing_left, consts):
        self.raw_state = raw_state.copy()
        self.swing_left = swing_left
        self.consts = consts

    def starting_platforms(self):
        # These should be returned in "chronologial" order: most recent platform last.
        return [self.swing_platform(), self.stance_platform()]

    def pose(self):
        return self.raw_state[ 0:-9]

    def dq(self):
        return self.raw_state[self.consts.Q_DIM:2*self.consts.Q_DIM]

    def stance_heel_location(self):
        return self.raw_state[-9:-6]

    def stance_platform(self):
        return self.raw_state[-6:-3]

    def swing_platform(self):
        return self.raw_state[-3:  ]

    def crashed(self):
        return not np.isfinite(self.raw_state).all()

    def rotate(self, angle):
        if self.consts.BRICK_DOF == 6:
            self.raw_state[self.consts.ROOT_YAW] += angle
            rot = utils.rotmatrix(-angle)
            self.pose()[:3] = np.dot(rot, self.pose()[:3])
            self.dq()[:3] = np.dot(rot, self.dq()[:3])
            self.stance_heel_location()[:3] = np.dot(rot, self.stance_heel_location()[:3])
            self.stance_platform()[:3] = np.dot(rot, self.stance_platform()[:3])
            self.swing_platform()[:3] = np.dot(rot, self.swing_platform()[:3])
            return rot

    def mirror(self):
        D = self.consts.LEG_DOF
        R = self.consts.RIGHT_IDX
        L = self.consts.LEFT_IDX
        for base in [0, self.consts.Q_DIM]:
            right = self.raw_state[base+R : base+R+D].copy()
            left  = self.raw_state[base+L : base+L+D].copy()
            self.raw_state[base+R : base+R+D] = left
            self.raw_state[base+L : base+L+D] = right
        self.swing_left = not self.swing_left

        if self.consts.BRICK_DOF == 6:
            # Rotations are absolute, not relative, so we need to multiply some angles
            # by -1 to obtain a mirrored pose.
            D = self.consts.Q_DIM
            m = np.ones(D)
            m[self.consts.absolute_rotation_indices] = -1
            self.raw_state[0:D] *= m
            self.raw_state[D:2*D] *= m
            self.raw_state[[-7,-4,-1]] *= -1 # Z coordinates of heel location and platforms

    def translate(self, new_origin):
        self.pose()[:3] -= new_origin
        self.stance_heel_location()[:3] -= new_origin
        self.stance_platform()[:3] -= new_origin
        self.swing_platform()[:3] -= new_origin

    def extract_features(self, target):
        # Combines state and target into a single vector, and discards any information
        # that does not affect the dynamics (heading, mirroring, and absolute location).
        # This vector is still very high dimensional (for debugging purposes).
        # When actually training the model, many of these features are discarded.
        copy = self.copy()
        target = target.copy()

        if self.swing_left:
            copy.mirror()
            target *= [1,1,-1]

        new_origin = copy.stance_platform().copy()
        if self.consts.BRICK_DOF == 6:
            # So it's still centered properly in the viewer later.
            # (The viewer doesn't currently adjust for Y coordinate.)
            new_origin[1] -= self.consts.GROUND_LEVEL
        copy.translate(new_origin)
        target -= new_origin

        # This actually loses some important information (namely target_heading
        # relative to the direction the robot's actually going). Until I can remove
        # the dependence of Simbicon3D on knowing the target direction, I'm just going
        # to keep that angle as part of the state (by removing these lines of code).
        if self.consts.BRICK_DOF == 6:
            angle = copy.raw_state[self.consts.ROOT_YAW]
            rot = copy.rotate(-angle)
            target[:3] = np.dot(rot, target[:3])

        return np.concatenate([copy.raw_state, target])

    def copy(self):
        return State(self.raw_state, self.swing_left, self.consts)

def reconstruct_state(features, consts):
    return State(features[:-3], False, consts), features[-3:]

def test_mirror_state(env):
    from time import sleep
    env.reset(random=0.5)
    env.set_rot_manual(np.pi/2)
    env.track_point = [0,0,0]
    obs = env.current_observation()
    env.pause(0.5)
    obs.mirror()
    env.reset(obs)
    env.pause(0.5)

def test_feature_extraction(env):
    from time import sleep
    obs = env.reset(random=0.5)
    obs.rotate(np.pi/2)
    obs = env.reset(obs)
    target = np.random.rand(3)
    env.sdf_loader.put_dot(target, 'target')
    env.track_point = [0,0,0]
    env.pause(0.5)
    features = obs.extract_features(target)
    obs, target = reconstruct_state(features, env.consts())
    env.reset(obs)
    env.sdf_loader.put_dot(target, 'target')
    env.pause(0.5)

if __name__ == "__main__":
    from simple_3D_env import Simple3DEnv
    from stepping_stones_env import SteppingStonesEnv
    #env = Simple3DEnv()
    env = SteppingStonesEnv()
    #test_mirror_state(env)
    test_feature_extraction(env)
    embed()
