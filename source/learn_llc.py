import numpy as np
from IPython import embed
from simbicon_3D import Simbicon3D

from consts_common3D import *
import simbicon_params as sp

MAX_SLIPPING = 0.02 # meters
REFERENCE_STEP_LENGTH = 0.1
import cma

def test(env, length, param_setting, r=None, n=8):
    GL = env.consts().GROUND_LEVEL
    seed = np.random.randint(100000)
    env.reset(seed=seed, video_save_dir=None, render=r) # TODO maybe randomize this a bit?
    env.sdf_loader.put_grounds([[-3.0, GL, 0]])
    prev_stance_heel = env.controller.stance_heel
    a = param_setting.reshape(-1)
    for i in range(n):
        l = length*0.5 if i == 0 else length
        t = np.array([prev_stance_heel[0]+l, GL, 0])
        obs, terminated = env.simulate(t, target_heading=0.0, action=a)
        if terminated or slipping(env, prev_stance_heel) > MAX_SLIPPING:
            break
        prev_stance_heel = env.controller.stance_heel
    dist = prev_stance_heel[0]
    print("Distance achieved:", dist)
    return -dist

def slipping(env, prev_stance_heel):
    c = env.controller
    curr_swing_heel = c.ik.forward_kine(c.swing_idx)
    slip_distance = np.linalg.norm(prev_stance_heel - curr_swing_heel)
    print("Slip:", slip_distance)
    return slip_distance

def learn(env, n_iters):
    def f(action, render=None):
        return test(env, REFERENCE_STEP_LENGTH, action, r=render)
    opzer = cma.CMA(f, np.zeros((sp.N_PARAMS,1)), 0.1, np.diag(sp.PARAM_SCALE**2))
    for i in range(n_iters):
        val = f(opzer.mean, render=4)
        print("Iteration", i, ":", val)
        opzer.iter()

if __name__ == "__main__":
    from darwin_env import DarwinEnv
    env = DarwinEnv(Simbicon3D)
    env.sdf_loader.ground_width = 8.0
    p = np.zeros(sp.N_PARAMS)
    #test(env, REFERENCE_STEP_LENGTH, p, r=8)
    learn(env, 10)
