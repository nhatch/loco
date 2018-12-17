from IPython import embed
import numpy as np
from sdf_loader import RED, GREEN, BLUE
from utils import heading_from_vector
import pydart2.utils.transformations as libtransform
import time

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
        heel_location = np.dot(foot.transform()[:3,:3], foot_centric_offset) + foot.com()
        return heel_location

    def heading(self, swing_idx):
        # We only ever use this in 3D
        c = self.env.consts()
        # The forward direction. Again, trial and error.
        foot_centric_offset = np.array([0.0, 0.0, -1.0])
        foot = self.get_bodynode(swing_idx, c.FOOT_BODYNODE_OFFSET)
        direction = np.dot(foot.transform()[:3,:3], foot_centric_offset)
        current_heading = self.env.controller.heading(self.env.get_x()[0])
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

    def inv_kine(self, target, swing_idx, verbose=False):
        r, theta = self.transform_frame(target, swing_idx, verbose)
        c = self.env.consts()

        # Handle out-of-reach targets by moving them within reach
        if (r > c.L_THIGH + c.L_SHIN):
            r = c.L_THIGH + c.L_SHIN
        if (r < np.abs(c.L_THIGH - c.L_SHIN)):
            r = np.abs(c.L_THIGH - c.L_SHIN)

        cos_knee_inner = (r**2 - c.L_THIGH**2 - c.L_SHIN**2) / (-2 * c.L_THIGH * c.L_SHIN)
        cos_hip = (c.L_SHIN**2 - r**2 - c.L_THIGH**2) / (-2 * r * c.L_THIGH)
        knee_inner = np.arccos(cos_knee_inner)
        knee = knee_inner - np.pi
        hip = np.arccos(cos_hip)
        hip = hip + theta
        return hip, knee

    def transform_frame(self, target, swing_idx, verbose=False):
        c = self.env.consts()
        thigh = self.get_bodynode(swing_idx, c.THIGH_BODYNODE_OFFSET)
        # Locate the hip joint
        thigh_centric_offset = np.array([0.0, 0.5*c.L_THIGH, 0.0])
        hip_location = np.dot(thigh.transform()[:3,:3], thigh_centric_offset) + thigh.com()
        if verbose:
            self.env.sdf_loader.put_dot(hip_location, 'hip_joint', color=RED)
            print("CENTER JOINT:", hip_location)
        # Move the target to the coordinate system centered at the hip joint
        # facing in the same direction as the pelvis.
        pelvis = self.robot_skeleton.bodynodes[c.PELVIS_BODYNODE_IDX]
        rot = np.linalg.inv(pelvis.transform())[:3,:3]
        egocentric_target = np.dot(rot, target - hip_location)
        # Ignore the Z coordinate for now (TODO?)
        x, y, _ = egocentric_target

        # Transform to polar coordinates
        r = np.sqrt(x**2 + y**2)
        phi = np.arcsin(x/r)
        if y > 0:
            phi = np.pi - phi
        if verbose:
            print("POLAR COORDINATES:", r, phi)
        return r, phi

    def test(self, right=True):
        c = self.env.consts()
        idx = c.RIGHT_IDX if right else c.LEFT_IDX
        self.test_inv_kine(idx)
        heel_loc = self.forward_kine(idx)
        self.env.sdf_loader.put_dot(heel_loc[:3], 'heel_loc', color=BLUE)
        self.pause()

    def pause(self, sec=1.5):
        # In 3D, give the viewer some time to rotate the environment and look around.
        # TODO have some kind of background thread do rendering, to avoid this hack
        # and make 3D environments easier to explore visually.
        FPS = 20
        n = int(FPS*sec) if self.env.is_3D else 1
        for i in range(n):
            self.env.render()
            time.sleep(1/FPS)

    def test_inv_kine(self, idx, planar=False):
        c = self.env.consts()
        agent = self.robot_skeleton
        brick_pose, target = self.gen_brick_pose(planar)

        q, _ = self.env.get_x()
        print("BRICK POSE:", brick_pose)
        q[:c.BRICK_DOF] = brick_pose
        agent.q = self.env.from_features(q)

        self.env.sdf_loader.put_dot(target, 'ik_target', color=GREEN)
        print("TARGET:", target)
        hip, knee = self.inv_kine(target, idx, verbose=True)
        q = self.inv_kine_pose(hip, knee)
        agent.q = self.env.from_features(q)
        self.env.render()
        print("Foot heading:", self.heading(idx), "Robot heading:", brick_pose[c.ROOT_YAW])

    def inv_kine_pose(self, hip, knee, is_right_foot=True):
        q, _ = self.env.get_x()
        c = self.env.consts()
        base = c.RIGHT_IDX if is_right_foot else c.LEFT_IDX
        q[base+c.HIP_PITCH] = hip
        q[base+c.KNEE] = knee
        return q

    def test_pointing(self):
        c = self.env.consts()
        basic = self.env.reset(random=0.0)
        swing_idx = c.LEFT_IDX
        thigh = self.get_bodynode(swing_idx, c.THIGH_BODYNODE_OFFSET)
        o1 = [0, 0.5,-0.5]
        o2 = [0,-0.5,-0.5]
        com = thigh.com().copy()
        t = thigh.transform().copy()
        d1 = np.dot(thigh.transform()[:3,:3], o1) + com
        d2 = np.dot(thigh.transform()[:3,:3], o2) + com

        self.pause(0.5)

        obs = self.env.reset(random=0.4)
        trot = np.linalg.inv(thigh.relative_transform())
        rot = np.dot(t, trot)
        euler = libtransform.euler_from_matrix(rot, 'ryzx')
        obs.raw_state[3:6] = [euler[1], euler[0], euler[2]]
        env.reset(obs, random=0)
        obs.raw_state[0:3] += com - thigh.com()
        env.reset(obs, random=0)

        d1_act = np.dot(thigh.transform()[:3,:3], o1) + thigh.com()
        d2_act = np.dot(thigh.transform()[:3,:3], o2) + thigh.com()
        error = np.linalg.norm(d1-d1_act) + np.linalg.norm(d2-d2_act)
        print("Error:", error)

        self.pause(1)

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
    #ik.test()
    ik.test_pointing()
    embed()
