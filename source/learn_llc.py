import numpy as np
from IPython import embed
from simbicon_3D import Simbicon3D

from consts_common3D import *
import simbicon_params as sp

MAX_SLIPPING = 0.02 # meters
REFERENCE_STEP_LENGTH = 0.1
import cma
import utils

controllable_params = utils.build_mask(sp.N_PARAMS,
        [
        sp.IK_GAIN,
        sp.TORSO_WORLD,
        sp.STANCE_HIP_ROLL_EXTRA,
        sp.STANCE_ANKLE_RELATIVE,
        sp.UP_IDX+sp.SWING_HIP_WORLD,
        sp.UP_IDX+sp.SWING_KNEE_RELATIVE,
        sp.UP_IDX+sp.STANCE_KNEE_RELATIVE,
        sp.DN_IDX+sp.SWING_HIP_WORLD,
        sp.DN_IDX+sp.SWING_KNEE_RELATIVE,
        sp.DN_IDX+sp.STANCE_KNEE_RELATIVE,
        ])

def test(env, length, param_setting, r=None, n=8):
    GL = env.consts().GROUND_LEVEL
    seed = np.random.randint(100000)
    env.reset(seed=seed, video_save_dir=None, render=r) # TODO maybe randomize this a bit?
    env.sdf_loader.put_grounds([[-3.0, GL, 0]])
    prev_stance_heel = env.controller.stance_heel
    penalty = 0.0
    for i in range(n):
        l = length*0.5 if i == 0 else length
        t = np.array([prev_stance_heel[0]+l, GL, 0])
        obs, terminated = env.simulate(t, target_heading=0.0, action=param_setting, put_dots=True)
        new_stance_heel = env.controller.stance_heel
        penalty += np.abs(new_stance_heel[0] - t[0])
        slip_dist = slipping(env, prev_stance_heel)
        penalty += slip_dist
        if terminated:
            penalty += 100.0
        if r is not None:
            print("Slip:", slip_dist)
        if terminated or slip_dist > MAX_SLIPPING:
            break
        prev_stance_heel = new_stance_heel
    dist = prev_stance_heel[0]
    if r is not None:
        print("Distance achieved:", dist, "Penalty:", penalty)
    return -dist + penalty

def slipping(env, prev_stance_heel):
    c = env.controller
    curr_swing_heel = c.ik.forward_kine(c.swing_idx)
    slip_distance = np.linalg.norm(prev_stance_heel - curr_swing_heel)
    return slip_distance

def init_opzer(env):
    def f(action, render=None):
        params = np.zeros(sp.N_PARAMS)
        params[controllable_params] = action.reshape(-1)
        return test(env, REFERENCE_STEP_LENGTH, params, r=render)
    init_mean = np.zeros((controllable_params.sum(),1))
    init_cov = np.diag(sp.PARAM_SCALE[controllable_params]**2) / 10
    opzer = cma.CMA(f, init_mean, 0.2, init_cov, extra_lambda=40)
    return opzer

means = []
vals = []

def learn(opzer, n_iters):
    for i in range(n_iters):
        val = opzer.f(opzer.mean, render=4)
        means.append(opzer.mean)
        vals.append(val)
        print("Iteration", i, ":", val)
        opzer.iter()

if __name__ == "__main__":
    from darwin_env import DarwinEnv
    env = DarwinEnv(Simbicon3D)
    env.sdf_loader.ground_width = 8.0
    p = np.zeros(len(controllable_params))
    #test(env, REFERENCE_STEP_LENGTH, p, r=8)
    opzer = init_opzer(env)
    learn(opzer, 3)
    embed()
