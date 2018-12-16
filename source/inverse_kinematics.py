from IPython import embed
import numpy as np
from sdf_loader import RED, GREEN, BLUE

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
        foot = self.get_foot(swing_idx)
        # TODO is it cheating to pull foot.com() directly from the environment?
        heel_location = np.dot(foot.transform()[:3,:3], foot_centric_offset) + foot.com()
        return heel_location

    def get_foot(self, swing_idx):
        c = self.env.consts()
        if swing_idx == c.RIGHT_IDX:
            swing_foot_idx = c.RIGHT_BODYNODE_IDX
        else:
            swing_foot_idx = c.LEFT_BODYNODE_IDX
        foot = self.robot_skeleton.bodynodes[swing_foot_idx]
        return foot

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
        bodynode_idx = c.RIGHT_THIGH_IDX if swing_idx == c.RIGHT_IDX else c.LEFT_THIGH_IDX
        thigh = self.robot_skeleton.bodynodes[bodynode_idx]
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
        # In 3D, give the viewer some time to rotate the environment and look around.
        # TODO have some kind of background thread do rendering, to avoid this hack
        # and make 3D environments easier to explore visually.
        import time
        n = 1 if c.BRICK_DOF == 3 else 30
        for i in range(n):
            self.env.render()
            time.sleep(0.05)

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

    def inv_kine_pose(self, hip, knee, is_right_foot=True):
        q, _ = self.env.get_x()
        c = self.env.consts()
        base = c.RIGHT_IDX if is_right_foot else c.LEFT_IDX
        q[base+c.HIP_PITCH] = hip
        q[base+c.KNEE] = knee
        return q

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
    #env = SteppingStonesEnv()
    env = Simple3DEnv()
    ik = InverseKinematics(env.robot_skeleton, env)
    ik.test()
    embed()
