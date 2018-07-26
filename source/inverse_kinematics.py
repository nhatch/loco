from IPython import embed
import numpy as np
from sdf_loader import RED, GREEN, BLUE

class InverseKinematics:
    def __init__(self, env, target=None):
        self.env = env
        self.target = target

    def controller(self, obs):
        # TODO: Maybe use obs rather than directly pulling agent state from environment (to avoid accusations of cheating)
        td, tf = self.transform_frame(self.target, 0.0)
        # Right foot step
        q = self.inv_kine_pose(td, tf, True)
        q[5] = 0
        q[6] = 0
        q[7] = 0
        q[8] = 0
        return q[3:]

    def forward_kine(self, swing_idx):
        agent = self.env.robot_skeleton
        c = self.env.consts()
        q = agent.q
        # Adding torso, hip, knee, and ankle angles gives the angle of the foot
        # relative to flat ground.
        foot_angle = q[c.THETA_IDX]+q[swing_idx+c.HIP_OFFSET]+q[swing_idx+c.KNEE_OFFSET]+q[swing_idx+c.ANKLE_OFFSET]
        if swing_idx == c.RIGHT_IDX:
            swing_foot_idx = c.RIGHT_BODYNODE_IDX
        else:
            swing_foot_idx = c.LEFT_BODYNODE_IDX
        foot_com = agent.bodynodes[swing_foot_idx].com()
        offset = -0.5 * c.L_FOOT * np.array([np.cos(foot_angle), np.sin(foot_angle), 0.0])
        offset[1] -= c.FOOT_RADIUS # So we get the *bottom* of the heel
        return foot_com + offset

    def test_forward_kine(self):
        c = self.env.consts()
        self.test_inv_kine()
        r_heel = self.forward_kine(c.RIGHT_IDX)
        self.env.sdf_loader.put_dot(r_heel[0:2], color=BLUE)
        self.env.render()

    def inv_kine(self, targ_down, targ_forward):
        c = self.env.consts()

        r = np.sqrt(targ_down**2 + targ_forward**2)
        cos_knee = (r**2 - c.L_LEG**2 - c.L_SHIN**2) / (2 * c.L_LEG * c.L_SHIN)
        tan_theta3 = targ_forward / targ_down
        cos_theta4 = (r**2 + c.L_LEG**2 - c.L_SHIN**2) / (2 * c.L_LEG * r)

        # Handle out-of-reach targets by clamping cosines to 1
        if cos_knee > 1:
            cos_knee = 1
        if cos_theta4 > 1:
            cos_theta4 = 1

        # We want knee angle to be < 0
        #print("COS_KNEE:", cos_knee)
        knee = - np.arccos(cos_knee)
        # The geometry makes more sense if theta4 is > 0
        theta4 = np.arccos(cos_theta4)
        # Since knee < 0, we add theta3 and theta4 to get hip angle
        hip = np.arctan(tan_theta3) + theta4
        return hip, knee

    def inv_kine_pose(self, targ_down, targ_forward, is_right_foot=True):
        agent = self.env.robot_skeleton
        hip, knee = self.inv_kine(targ_down, targ_forward)
        q = agent.q.copy()
        c = self.env.consts()
        base = c.RIGHT_IDX if is_right_foot else c.LEFT_IDX
        q[base+c.HIP_OFFSET] = hip
        q[base+c.KNEE_OFFSET] = knee
        return q

    def test_inv_kine(self):
        self.env.clear_skeletons()
        self.env.reset(random=0.0)
        c = self.env.consts()
        agent = self.env.robot_skeleton
        # TODO: this test actually fails for targets above waist level.
        #   brick_pose = [-0.12118236,  0.13867071,  0.00750066]
        #   target = [-0.416734251601, 1.24994977515, 0] (after adding 0.5)
        brick_pose, target = self.gen_brick_pose()
        q = agent.q
        print("BRICK POSE:", brick_pose)
        q[:c.BRICK_DOF] = brick_pose
        agent.q = q
        self.env.sdf_loader.put_dot(target[0:2], color=GREEN)
        print("TARGET:", target[0], target[1])
        down, forward = self.transform_frame(target[0], target[1], verbose=True)
        q = self.inv_kine_pose(down, forward)
        agent.q = q
        self.env.render()

    def gen_brick_pose(self, planar=True):
        # Generate a random starting pose and target.
        D = self.env.consts().BRICK_DOF
        center = np.zeros(D)
        center[1] += 0.3
        brick_pose = center + np.random.uniform(low=-0.2, high=0.2, size=D)
        if planar and D == 6:
            non_planar_dofs = [2,4,5]
            brick_pose[non_planar_dofs] = 0.0
        target = center + np.random.uniform(low=-0.5, high=0.5, size=D)
        target[1] += 0.5
        return brick_pose, target

    def transform_frame(self, x, y, verbose=False):
        c = self.env.consts()
        agent = self.env.robot_skeleton
        # Takes the absolute coordinates (x,y) and transforms them into (down, forward)
        # relative to the pelvis joint's absolute location and rotation.
        pelvis_com = agent.bodynodes[c.PELVIS_BODYNODE_IDX].com()
        theta = agent.q[c.THETA_IDX]
        pelvis_bottom = pelvis_com + c.L_PELVIS/2 * np.array([np.sin(theta), -np.cos(theta), 0])
        if verbose:
            self.env.sdf_loader.put_dot(pelvis_bottom[:2], color=RED)
            print("PELVIS COM:", pelvis_com[0], pelvis_com[1])
        # Put pelvis bottom at origin
        x = x - pelvis_bottom[0]
        y = y - pelvis_bottom[1]
        # Transform to polar coordinates
        r = np.sqrt(x**2 + y**2)
        phi = np.arcsin(x/r)
        # Rotate
        phi = phi - theta
        if y > 0:
            phi = np.pi - phi
        if verbose:
            print("ANGLES:", phi, theta)
        # Transform back to Euclidean coordinates
        down = r * np.cos(phi)
        forward = r * np.sin(phi)
        if verbose:
            print("RELATIVE TO JOINT:", down, forward)
        return down, forward


if __name__ == "__main__":
    #from stepping_stones_env import SteppingStonesEnv
    #env = SteppingStonesEnv()
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv()
    ik = InverseKinematics(env, 0.5)
    ik.test_forward_kine()
    embed()
