from IPython import embed
import numpy as np
from sdf_loader import RED, GREEN, BLUE

class InverseKinematics:
    def __init__(self, env, target=None):
        self.env = env
        self.target = target

    def controller(self, obs):
        # TODO: Maybe use obs rather than directly pulling agent state from environment (to avoid accusations of cheating)
        hip, knee = self.inv_kine([self.target, 0.0])
        # Right foot step
        q = self.inv_kine_pose(hip, knee, True)
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

    def inv_kine(self, target, verbose=False):
        r, theta = self.transform_frame(target, verbose)
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

    def inv_kine_pose(self, hip, knee, is_right_foot=True):
        agent = self.env.robot_skeleton
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
        brick_pose, target = self.gen_brick_pose()
        q = agent.q
        print("BRICK POSE:", brick_pose)
        q[:c.BRICK_DOF] = brick_pose
        agent.q = q
        self.env.sdf_loader.put_dot(target[0:2], color=GREEN)
        print("TARGET:", target)
        hip, knee = self.inv_kine(target, verbose=True)
        q = self.inv_kine_pose(hip, knee)
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

    def transform_frame(self, target, verbose=False):
        c = self.env.consts()
        agent = self.env.robot_skeleton
        # Takes the absolute coordinates (x,y) and transforms them into (down, forward)
        # relative to the pelvis joint's absolute location and rotation.
        pelvis_com = agent.bodynodes[c.PELVIS_BODYNODE_IDX].com()
        theta = agent.q[c.THETA_IDX]
        pelvis_bottom = pelvis_com + c.L_PELVIS/2 * np.array([np.sin(theta), -np.cos(theta), 0])
        if verbose:
            self.env.sdf_loader.put_dot(pelvis_bottom[:2], color=RED)
            print("CENTER JOINT:", pelvis_bottom)
        # Put pelvis bottom at origin
        x = target[0] - pelvis_bottom[0]
        y = target[1] - pelvis_bottom[1]
        # Transform to polar coordinates
        r = np.sqrt(x**2 + y**2)
        phi = np.arcsin(x/r)
        # Rotate
        phi = phi - theta
        if y > 0:
            phi = np.pi - phi
        if verbose:
            print("POLAR COORDINATES:", r, phi)
        return r, phi

if __name__ == "__main__":
    #from stepping_stones_env import SteppingStonesEnv
    #env = SteppingStonesEnv()
    from simple_3D_env import Simple3DEnv
    env = Simple3DEnv()
    ik = InverseKinematics(env, 0.5)
    ik.test_forward_kine()
    embed()
