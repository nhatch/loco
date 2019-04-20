from IPython import embed
import numpy as np
from sdf_loader import RED, GREEN, BLUE
from utils import heading_from_vector
import libtransform

class InverseKinematics:
    def __init__(self, skel, env):
        self.robot_skeleton = skel
        self.env = env

    def forward_kine(self, swing_idx):
        c = self.env.consts()
        if not hasattr(self.env, 'world'):
            return np.zeros(3)
        if c.BRICK_DOF == 3:
            foot_centric_offset = np.array([-0.5*c.L_FOOT, -c.FOOT_RADIUS, 0.0])
        elif c.GRAVITY_Y:
            # I found the signs/order of these offsets by trial and error.
            # I think the reason it's different might be the axis_order setting
            # of 'zyx' in the skel file for the simple 3D model.
            foot_centric_offset = np.array([0.0, -c.FOOT_RADIUS, 0.5*c.L_FOOT])
        else:
            # Z-axis gravity. Again, trial and error
            foot_centric_offset = np.array([c.FOOT_RADIUS, 0.0, -0.5*c.L_FOOT])
            if swing_idx == c.LEFT_IDX:
                foot_centric_offset[0] *= -1
        foot = self.get_bodynode(swing_idx, c.FOOT_BODYNODE_OFFSET)
        # TODO is it cheating to pull foot.com() directly from the environment?
        heel_location = np.dot(foot.transform()[:3,:3], foot_centric_offset) + foot.com()
        return c.inverse_convert_root(heel_location)

    def get_bodynode(self, swing_idx, offset):
        c = self.env.consts()
        if swing_idx == c.RIGHT_IDX:
            swing_side_idx = c.RIGHT_BODYNODE_IDX
        else:
            swing_side_idx = c.LEFT_BODYNODE_IDX
        bodynode = self.robot_skeleton.bodynodes[swing_side_idx+offset]
        return bodynode

    def root_bodynode(self):
        c = self.env.consts()
        return self.robot_skeleton.bodynodes[c.PELVIS_BODYNODE_IDX]

    def test(self, right=True):
        c = self.env.consts()
        idx = c.RIGHT_IDX if right else c.LEFT_IDX
        self.env.reset(random=0.3)
        heel_loc = self.forward_kine(idx)
        self.env.sdf_loader.put_dot(heel_loc[:3], 'heel_loc', color=BLUE)
        self.env.pause()

    # Returns the values of the 6 "brick DOFs" for which, if the rest of the DOFs
    # remain the same, the given bodynode will have the given target transformation.
    def get_dofs(self, target_transform, bodynode):
        current_transform = bodynode.transform()
        # TODO if the root_bodynode().transform() is not the identity when
        # all DOFs of the robot are zero, this is wrong.
        # For the 2D model, I've hacked around this problem using root_dofs_from_transform.
        base_transform = self.root_bodynode().transform()
        relative_transform = np.linalg.inv(base_transform).dot(current_transform)
        target_base_transform = target_transform.dot(np.linalg.inv(relative_transform))
        c = self.env.consts()
        return c.root_dofs_from_transform(target_base_transform)

    def test_inverse_transform(self, bodynode):
        # TODO for Darwin, this seems to be broken for certain bodynodes (e.g. 10, 14)

        self.env.reset(random=0.4)
        c = self.env.consts()
        # First just test whether `root_dofs_from_transform` is working correctly.
        inferred_root_dofs = c.root_dofs_from_transform(self.root_bodynode().transform())
        true_root_dofs = env.current_observation().raw_state[:6]
        print(np.allclose(true_root_dofs, inferred_root_dofs))

        orig_transform = bodynode.transform()
        self.env.pause(0.5)

        # Now try to match that absolute transform with a different joint configuration
        obs = self.env.reset(random=0.4)
        obs.raw_state[0:c.BRICK_DOF] = self.get_dofs(orig_transform, bodynode)
        env.reset(obs)
        print(libtransform.is_same_transform(orig_transform, bodynode.transform()))
        self.env.pause(0.5)

    # `bodynode` must be one of the thighs. Returns the Euler angles for that hip joint
    # such that, if the transform of that thigh stays fixed, then the pelvis orientation
    # will match the orientation of `target_root_transform`. (Note however that the
    # translation of `target_root_transform` will not in general be achieved.)
    def get_hip(self, stance_idx, target_root_transform):
        c = self.env.consts()
        thigh = self.get_bodynode(stance_idx, c.THIGH_BODYNODE_OFFSET)
        thigh_transform = thigh.transform()
        target_relative_transform = np.linalg.inv(target_root_transform).dot(thigh_transform)
        return self.hip_dofs_from_transform(target_relative_transform, stance_idx)

    def hip_dofs_from_transform(self, hip_relative_transform, stance_idx):
        c = self.env.consts()
        if stance_idx == c.LEFT_IDX:
            RRT = c.LEFT_RRT_INV
        else:
            RRT = c.RIGHT_RRT_INV
        target_dof_transform = RRT.dot(hip_relative_transform)
        return c.hip_dofs_from_transform(stance_idx, target_dof_transform)

    # Gives the transform corresponding to the given heading and pitch (with zero roll)
    def root_transform_from_angles(self, heading, pitch):
        c = self.env.consts()
        if c.GRAVITY_Y:
            return libtransform.euler_matrix(0.0, heading, pitch, 'rxyz')
        else:
            return libtransform.euler_matrix(heading, -pitch, 0.0, 'rzyx')

    def get_ankle(self, stance_idx, pitch=0.0, roll=0.0, target_foot_transform=None):
        c = self.env.consts()
        if target_foot_transform is None:
            target_foot_transform = c.foot_transform_from_angles(stance_idx, pitch, roll)
        shin = self.get_bodynode(stance_idx, c.SHIN_BODYNODE_OFFSET)
        target_rel_trans = np.linalg.inv(shin.transform()).dot(target_foot_transform)
        if stance_idx == c.RIGHT_IDX:
            RRT = c.RIGHT_RRT_INV_ANKLE
        else:
            RRT = c.LEFT_RRT_INV_ANKLE
        return c.ankle_dofs_from_transform(stance_idx, RRT.dot(target_rel_trans))

    def test_inverse_ankle(self, stance_idx, pitch=0.0, roll=0.0):
        c = self.env.consts()
        obs = self.env.reset(random=0.4)
        ANKLE_DOF = 2
        if not self.env.is_3D:
            roll = 0.0
            ANKLE_DOF = 1

        foot = self.get_bodynode(stance_idx, c.FOOT_BODYNODE_OFFSET)
        inferred_ankle_dofs = self.get_ankle(stance_idx, target_foot_transform=foot.transform())
        print(inferred_ankle_dofs)
        true_ankle_dofs = obs.raw_state[stance_idx+c.ANKLE:stance_idx+c.ANKLE+ANKLE_DOF]
        print(true_ankle_dofs)
        print(np.allclose(inferred_ankle_dofs, true_ankle_dofs, atol=1e-7))

        ground_loc = c.inverse_convert_root(foot.com()) - [0,0.01,0]
        env.sdf_loader.put_grounds([ground_loc])
        self.env.pause(0.5)
        ankle_dofs = self.get_ankle(stance_idx, pitch, roll)
        obs.raw_state[stance_idx+c.ANKLE:stance_idx+c.ANKLE+ANKLE_DOF] = ankle_dofs

        env.reset(obs)
        env.sdf_loader.put_grounds([ground_loc])
        self.env.pause(0.5)

    def test_inverse_hip(self, stance_idx, heading=0.0, pitch=0.0):
        c = self.env.consts()
        obs = self.env.reset(random=0.4)

        HIP_DOF = 3
        if not self.env.is_3D:
            heading = 0.0
            HIP_DOF = 1

        thigh = self.get_bodynode(stance_idx, c.THIGH_BODYNODE_OFFSET)
        pelvis = self.root_bodynode()

        # First just test get_hip with the current root transform.
        # (Should just return the current hip DOFs.)
        inferred_hip_dofs = self.get_hip(stance_idx, pelvis.transform())
        print(inferred_hip_dofs)
        true_hip_dofs = obs.raw_state[stance_idx:stance_idx+HIP_DOF]
        print(true_hip_dofs)
        print(np.allclose(inferred_hip_dofs, true_hip_dofs, atol=1e-7))

        target_root_transform = self.root_transform_from_angles(heading, pitch)
        orig_thigh_transform = thigh.transform()
        self.env.pause(0.5)

        # Now use the hip to point the pelvis in the target direction
        hip_dofs = self.get_hip(stance_idx, target_root_transform)
        obs.raw_state[stance_idx:stance_idx+HIP_DOF] = hip_dofs
        env.reset(obs)
        # Rotate the whole robot so the transform of bodynode doesn't change
        obs.raw_state[0:c.BRICK_DOF] = self.get_dofs(orig_thigh_transform, thigh)
        env.reset(obs)
        # We can't get the same translation (3 DOFs vs 6 DOFs), but the orientation
        # should be correct.
        # For some reason there seems to be some numerical error sometimes
        print(np.allclose(target_root_transform[:3,:3], pelvis.transform()[:3,:3], atol=1e-5))
        self.env.pause(0.5)

if __name__ == "__main__":
    from simbicon_3D import Simbicon3D
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from darwin_env import DarwinEnv
    env = SteppingStonesEnv()
    #env = Simple3DEnv(Simbicon3D)
    #env = DarwinEnv(Simbicon3D)
    env.track_point = [0,0,0]
    ik = InverseKinematics(env.robot_skeleton, env)
    #ik.test(False)
    c = env.consts()
    bodynode = env.robot_skeleton.bodynodes[c.RIGHT_BODYNODE_IDX+c.THIGH_BODYNODE_OFFSET]
    #ik.test_inverse_transform(bodynode)
    #ik.test_inverse_hip(c.RIGHT_IDX, heading=-1.0, pitch=0.2)
    ik.test_inverse_ankle(c.RIGHT_IDX, pitch=0.0, roll=0.0)
    embed()
