from IPython import embed
import numpy as np

L_PELVIS = 0.40
L_LEG =    0.45
L_SHIN =   0.50
L_FOOT =   0.20

class InverseKinematics:
    def __init__(self, env, target=None):
        self.env = env
        self.agent = self.env.robot_skeleton
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

    def forward_kine(self, swing_idx=3):
        q = self.agent.q
        # Adding torso, hip, knee, and ankle angles gives the angle of the foot
        # relative to flat ground.
        foot_angle = q[2]+q[swing_idx]+q[swing_idx+1]+q[swing_idx+2]
        foot_com = self.agent.bodynodes[swing_idx+2].com()
        offset = -0.5 * L_FOOT * np.array([np.cos(foot_angle), np.sin(foot_angle), 0.0])
        return foot_com + offset

    def test_forward_kine(self):
        self.test_inv_kine()
        r_heel = self.forward_kine()
        # If something's broken, this will move the dot away from the ankle
        self.env.put_dot(r_heel[0], r_heel[1])
        self.env.render()

    def inv_kine(self, targ_down, targ_forward):

        r = np.sqrt(targ_down**2 + targ_forward**2)
        cos_knee = (r**2 - L_LEG**2 - L_SHIN**2) / (2 * L_LEG * L_SHIN)
        tan_theta3 = targ_forward / targ_down
        cos_theta4 = (r**2 + L_LEG**2 - L_SHIN**2) / (2 * L_LEG * r)

        # Handle out-of-reach targets by clamping cosines to 1
        if cos_knee > 1:
            cos_knee = 1
        if cos_theta4 > 1:
            cos_theta4 = 1

        # We want knee angle to be < 0
        knee = - np.arccos(cos_knee)
        # The geometry makes more sense if theta4 is > 0
        theta4 = np.arccos(cos_theta4)
        # Since knee < 0, we add theta3 and theta4 to get hip angle
        hip = np.arctan(tan_theta3) + theta4
        return hip, knee

    def inv_kine_pose(self, targ_down, targ_forward, is_right_foot=True):
        hip, knee = self.inv_kine(targ_down, targ_forward)
        q = self.agent.q.copy()
        if is_right_foot:
            q[3] = hip
            q[4] = knee
        else: # left foot
            q[6] = hip
            q[7] = knee
        return q

    def test_inv_kine(self):
        # Generate a random starting pose and target (marked with a dot),
        # then edit the right hip and knee joints to hit the target.
        #
        # TODO: this test actually fails for:
        #   pose = [-0.12118236,  0.13867071,  0.00750066]
        #   target = [-0.416734251601, 1.24994977515, 0] (after adding 0.5)
        center = np.array([0, 0.3, 0])
        pose = center + np.random.uniform(low=-0.2, high = 0.2, size = 3)
        q = self.agent.q
        q[:3] = pose
        self.agent.q = q
        target = center + np.random.uniform(low=-0.5, high=0.5, size=3)
        target[1] += 0.5
        print("TARGET:", target[0], target[1])
        down, forward = self.transform_frame(target[0], target[1], verbose=True)
        q = self.inv_kine_pose(down, forward)
        self.agent.q = q
        self.env.render()

    def transform_frame(self, x, y, verbose=False):
        # Takes the absolute coordinates (x,y) and transforms them into (down, forward)
        # relative to the pelvis joint's absolute location and rotation.
        pelvis_com = self.agent.bodynodes[2].com()
        theta = self.agent.q[2]
        self.env.put_dot(x, y)
        if verbose:
            print("PELVIS COM:", pelvis_com[0], pelvis_com[1])
        # Put pelvis COM at origin
        x = x - pelvis_com[0]
        y = y - pelvis_com[1]
        # Transform to polar coordinates
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan(y/x)
        if phi*y < 0:
            phi = phi - np.pi
        if verbose:
            print("ANGLES:", phi, theta)
        # Rotate
        phi = phi - theta
        # Transform back to Euclidean coordinates
        down = -r * np.sin(phi)
        forward = r * np.cos(phi)
        # Use bottom of pelvis instead of COM
        down = down - L_PELVIS / 2
        if verbose:
            print("RELATIVE TO JOINT:", down, forward)
        return down, forward


if __name__ == "__main__":
    from random_search import Whitener
    import stepper
    import gym

    env = gym.make('Stepper-v0')
    ik = InverseKinematics(env.env, 0.5)
    #ik.test_inv_kine()
    ik.test_forward_kine()
    embed()
