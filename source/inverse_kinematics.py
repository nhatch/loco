from IPython import embed
import numpy as np
from sdf_loader import RED, GREEN, BLUE
from utils import heading_from_vector
import pydart2.utils.transformations as libtransform

class InverseKinematics:
    def __init__(self, skel, env):
        self.robot_skeleton = skel
        self.env = env

    def forward_kine(self, swing_idx):
        c = self.env.consts()
        if c.BRICK_DOF == 3:
            foot_centric_offset = np.array([-0.5*c.L_FOOT, -c.FOOT_RADIUS, 0.0])
        else:
            # I found the signs/order of these offsets by trial and error.
            # I think the reason it's different might be the axis_order setting
            # of 'zyx' in the skel file for the simple 3D model.
            foot_centric_offset = np.array([0.0, -c.FOOT_RADIUS, 0.5*c.L_FOOT])
        foot = self.get_bodynode(swing_idx, c.FOOT_BODYNODE_OFFSET)
        # TODO is it cheating to pull foot.com() directly from the environment?
        # TODO fix this for Z gravity!
        heel_location = np.dot(foot.transform()[:3,:3], foot_centric_offset) + foot.com()
        return heel_location

    def heading(self, swing_idx):
        # We only ever use this in 3D
        c = self.env.consts()
        # The forward direction. Again, trial and error.
        foot_centric_offset = np.array([0.0, 0.0, -1.0])
        foot = self.get_bodynode(swing_idx, c.FOOT_BODYNODE_OFFSET)
        direction = np.dot(foot.transform()[:3,:3], foot_centric_offset)
        current_heading = self.env.controller.heading()
        _, heading = heading_from_vector(direction, current_heading)
        return heading

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
        if self.env.is_3D:
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
        self.env.reset(random=0.4)
        orig_transform = bodynode.transform()
        self.env.pause(0.5)

        obs = self.env.reset(random=0.4)
        c = self.env.consts()
        obs.raw_state[0:c.BRICK_DOF] = self.get_dofs(orig_transform, bodynode)
        env.reset(obs)
        print(libtransform.is_same_transform(orig_transform, bodynode.transform()))
        self.env.pause(0.5)

    # `bodynode` must be one of the thighs. Returns the Euler angles for that hip joint
    # such that, if the transform of that thigh stays fixed, then the pelvis orientation
    # will match the orientation of `target_root_transform`. (Note however that the
    # translation of `target_root_transform` will not in general be achieved.)
    def get_hip(self, bodynode, target_root_transform):
        thigh_transform = bodynode.transform()
        c = self.env.consts()
        if bodynode.id == c.LEFT_BODYNODE_IDX + c.THIGH_BODYNODE_OFFSET:
            RRT = c.LEFT_RRT
        else:
            RRT = c.RIGHT_RRT
        target_relative_transform = np.linalg.inv(target_root_transform).dot(thigh_transform)
        target_dof_transform = RRT.dot(target_relative_transform)
        return c.hip_dofs_from_transform(target_dof_transform)

    # Gives the transform corresponding to the given heading and pitch (with zero roll)
    def root_transform_from_angles(self, heading, pitch):
        return libtransform.euler_matrix(0.0, heading, pitch, 'rxyz')

    def test_inverse_hip(self, swing_idx, heading=0.0, pitch=0.0):
        if not self.env.is_3D:
            heading = 0.0
        target_root_transform = self.root_transform_from_angles(heading, pitch)
        obs = self.env.reset(random=0.4)
        c = self.env.consts()
        thigh = self.get_bodynode(swing_idx, c.THIGH_BODYNODE_OFFSET)
        orig_thigh_transform = thigh.transform()
        self.env.pause(0.5)

        # Use the hip to point the pelvis in the target direction
        hip_dofs = self.get_hip(thigh, target_root_transform)
        if self.env.is_3D:
            obs.raw_state[swing_idx:swing_idx+3] = hip_dofs
        else:
            obs.raw_state[swing_idx] = hip_dofs
        env.reset(obs)
        # Rotate the whole robot so the transform of bodynode doesn't change
        obs.raw_state[0:c.BRICK_DOF] = self.get_dofs(orig_thigh_transform, thigh)
        env.reset(obs)
        pelvis = self.root_bodynode()
        # We can't get the same translation (3 DOFs vs 6 DOFs), but the orientation
        # should be correct.
        # For some reason there seems to be some numerical error sometimes
        print(np.allclose(target_root_transform[:3,:3], pelvis.transform()[:3,:3], atol=1e-6))
        self.env.pause(0.5)

    def gen_brick_pose(self, planar):
        # Generate a random starting pose and target.
        D = self.env.consts().BRICK_DOF
        center = np.zeros(D)
        if D == 3:
            scale = [0.2, 0.2, 0.2]
        else:
            scale = [0.2, 0.2, 0.2, np.pi/6, 2*np.pi, np.pi/6]
            if planar:
                non_planar_dofs = [2,4,5]
                scale[non_planar_dofs] = 0.0
        brick_pose = center + scale*np.random.uniform(low=-0.5, high=0.5, size=D)
        target = center + np.random.uniform(low=-0.5, high=0.5, size=D)
        if D == 3:
            target[1] += 0.5
            # Technically 2 represents the pitch, but here we're repurposing it as Z coordinate
            target[2] = 0.0
        else:
            target[1] -= 0.5
        return brick_pose, target[:3]

if __name__ == "__main__":
    from stepping_stones_env import SteppingStonesEnv
    from simple_3D_env import Simple3DEnv
    from simbicon_3D import Simbicon3D
    #env = SteppingStonesEnv()
    env = Simple3DEnv(Simbicon3D)
    env.track_point = [0,0,0]
    ik = InverseKinematics(env.robot_skeleton, env)
    ik.test()
    embed()
    c = env.consts()
    bodynode = env.robot_skeleton.bodynodes[c.RIGHT_BODYNODE_IDX+c.THIGH_BODYNODE_OFFSET]
    #ik.test_inverse_transform(bodynode)
    ik.test_inverse_hip(c.RIGHT_IDX, heading=-1.0, pitch=0.2)
    embed()
