import gym
import time
import numpy as np
import sys
from IPython import embed

class PDStand:
    def __init__(self):
        self.env = gym.make('DartHopper-v1')
        self.n_act = self.env.action_space.shape[0]

    def flop(self):
        self.env.reset()
        while True:
            control = np.zeros(self.n_act)
            observation, reward, done, info = self.env.step(control)
            self.env.render()
            if done:
                return

    def simulate(self, controller, steps):
        for _ in range(steps):
            self.obs, _, _, _ = self.env.step(controller)
            self.env.render()
        return self.obs

    def stand(self):
        obs = self.env.reset()
        pose = obs[2:5] # hip, knee, and ankle angles (only DOFs we control)
        target_pose = np.array([-0.05,-0.08,0.05])*np.pi # slightly bent joints for stability?
                                                         # and to avoid hitting joint limits?
        prev_pose = pose
        prev_d = np.zeros(self.n_act)
        for i in range(1000):
            if i % 200 == 100:
                q = self.env.env.robot_skeleton.q.copy()
                dq = self.env.env.robot_skeleton.dq.copy()
                # Edit these as desired, to introduce perturbations
                dq[0] += i/200
                dq[1] += i/200
                self.env.env.set_state(q, dq)
            control, prev_d = self.pd_control(pose, target_pose, prev_pose, prev_d)
            obs, _, done, _ = self.env.step(control)
            prev_pose, pose = pose, obs[2:5]
            self.env.render()
            if done:
                pass
                #return

    def pd_control(self, p, target_p, prev_p, prev_d):
        # Tuned these constants via guess and check.
        kp = np.array([1,1,1])
        kd = np.sqrt(2*kp) + [0,0,3]
        d_damper = 0.8
        diff = p - prev_p
        d = d_damper*prev_d + (1-d_damper)*diff
        control = -kp * (p - target_p) - kd * d
        return control, d

if __name__ == "__main__":
    #PDStand().flop()
    PDStand().stand()
