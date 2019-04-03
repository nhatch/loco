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
        #sp.IK_GAIN,
        sp.UP_DURATION,
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

def test(env, length, param_setting, render=None, n=100, terminate_on_slip=True):
    GL = env.consts().GROUND_LEVEL
    seed = np.random.randint(100000)
    env.reset(seed=seed, random=0.005, video_save_dir=None, render=render)
    env.sdf_loader.put_grounds([[-3.0, GL, 0]])
    prev_stance_heel = env.controller.stance_heel
    t = prev_stance_heel.copy()
    t[2] = 0.0
    penalty = 0.0
    for i in range(n):
        l = length*0.5 if i == 0 else length
        #t[0] = prev_stance_heel[0]
        t[0] += l
        obs, terminated = env.simulate(t, target_heading=0.0, action=param_setting, put_dots=True)
        if terminated:
            penalty += 100.0
        new_stance_heel = env.controller.stance_heel
        penalty += np.abs(new_stance_heel[0] - t[0])
        slip_dist = slipping(env, prev_stance_heel)
        if render is not None:
            print("Slip:", slip_dist)
        penalty += slip_dist
        if slip_dist > MAX_SLIPPING and terminate_on_slip:
            terminated = True
        if terminated:
            break
        prev_stance_heel = new_stance_heel
    dist = prev_stance_heel[0]
    if render is not None:
        print("Distance achieved:", dist, "Penalty:", penalty)
    return -dist + penalty

def slipping(env, prev_stance_heel):
    c = env.controller
    curr_swing_heel = c.ik.forward_kine(c.swing_idx)
    slip_distance = np.linalg.norm(prev_stance_heel - curr_swing_heel)
    return slip_distance

def init_opzer(env, init_mean):
    def f(action, **kwargs):
        params = np.zeros(sp.N_PARAMS)
        params[controllable_params] = action.reshape(-1)
        return test(env, REFERENCE_STEP_LENGTH, params, **kwargs)
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

b0 = np.zeros((controllable_params.sum(),1))

# Best so far: observes target
# Works only if you check out commit 6e2c0667db5df3b42548ac8b14a47aeea5929826
# (use these same numbers though)
b2 = np.array([[-0.18549841],
       [-0.06580528],
       [ 0.15634049],
       [-0.37124493],
       [ 0.97356754],
       [-0.05797221],
       [-0.19359471]])

# Best so far: does not observe target (problem: knees don't obey joint limits)
b3 = np.array([[-0.12618084],
       [ 0.227121  ],
       [-0.09054774],
       [ 0.35178088],
       [ 0.27585969],
       [-0.1552619 ],
       [ 0.63135911],
       [ 0.01060128],
       [ 0.07810596],
       [-0.01016673]])


if __name__ == "__main__":
    from darwin_env import DarwinEnv
    env = DarwinEnv(Simbicon3D)
    env.sdf_loader.ground_width = 8.0
    p = np.zeros(len(controllable_params))
    #test(env, REFERENCE_STEP_LENGTH, p, r=8)
    opzer = init_opzer(env, b0)
    opzer.f(b3, render=2, terminate_on_slip=False)
    #learn(opzer, 40)
    embed()
