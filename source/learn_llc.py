import numpy as np
from IPython import embed

from consts_common3D import *
import simbicon_params as sp

REFERENCE_STEP_LENGTH = 0.1
import cma
import utils

controllable_params = utils.build_mask(sp.N_PARAMS,
        [
        sp.TORSO_WORLD,
        sp.STANCE_HIP_ROLL_EXTRA,
        sp.STANCE_ANKLE_RELATIVE,
        sp.UP_IDX+sp.SWING_HIP_WORLD,
        sp.UP_IDX+sp.SWING_KNEE_RELATIVE,
        sp.UP_IDX+sp.STANCE_KNEE_RELATIVE,
        sp.POSITION_BALANCE_GAIN,
        sp.VELOCITY_BALANCE_GAIN,
        sp.UP_DURATION,
        ])

def test(env, length, param_setting, render=None, n=50, terminate_on_slip=True):
    GL = env.consts().GROUND_LEVEL
    seed = np.random.randint(100000)
    env.reset(seed=seed, random=0.005, video_save_dir=None, render=render)
    env.sdf_loader.put_grounds([[-3.0, GL, 0]])
    c = env.controller
    penalty = 0.0
    for i in range(n):
        obs, terminated, n_float = env.simulate([i*length, GL, 0], target_heading=0.0,
                action=param_setting,
                put_dots=True, count_float=True)
        if terminated:
            penalty += 100.0
        if render is not None:
            print("n_float:", n_float)
        penalty += n_float/100
        if n_float > 2 and i > 0 and terminate_on_slip:
            terminated = True
        if terminated:
            break
    dist = c.stance_heel[0]
    penalty += np.abs(dist - i*length)
    penalty += np.abs(obs.pose()[env.consts().ROOT_PITCH]) # Stay upright
    penalty += n_float / 100
    if render is not None:
        print("Distance achieved:", dist, "Penalty:", penalty)
    return penalty - dist#i*length # AAAAAAAAAAGH

def embed_action(action):
    params = np.zeros(sp.N_PARAMS)
    params[controllable_params] = action.reshape(-1)
    return params

def init_opzer(env, init_mean):
    def f(action, **kwargs):
        return test(env, REFERENCE_STEP_LENGTH, embed_action(action), **kwargs)
    init_cov = np.diag(sp.PARAM_SCALE[controllable_params]**2) / 10
    opzer = cma.CMA(f, init_mean, 0.2, init_cov, extra_lambda=40)
    return opzer

means = []
vals = []

def learn(opzer, n_iters):
    try:
        for i in range(n_iters):
            val = opzer.f(opzer.mean, render=2)
            means.append(opzer.mean)
            vals.append(val)
            graph(vals, opzer)
            print("Iteration", i, ":", val)
            opzer.iter()
    except SystemExit: # Pydart for some dumb reason turns keyboard interrupts into this
        return

def graph(vals, opzer):
    import matplotlib.pyplot as plt
    plt.plot(vals[1:])
    root = 'data/learn_llc/'
    plt.savefig(root+'learning_curve.pdf')
    plt.clf()
    np.savetxt(root+'mean.txt', opzer.mean)

b0 = np.zeros((controllable_params.sum(),1))

# Tends to wander off to one side, but otherwise seems stable.
b2 = np.array([
3.481975631115105663e-01,
-3.501715375626047178e-01,
-1.052174403778687040e-01,
1.403727068552419743e-01,
2.621220874820474056e-01,
-4.264195414790594718e-01,
-8.261033834150999233e-04,
-6.957290456205512258e-03,
0.
]).reshape((-1, 1))

# Best gait in simulation so far, but it walks (almost runs actually) on
# its toes, so might have a significant reality gap.
b3 = np.array([
-2.957959711096587618e+00,
8.040670712779093066e+00,
-4.122368782998938053e-01,
8.294588160103437413e-02,
6.341520027721733177e-01,
-6.218221083543284955e-01,
-4.472706620440605185e-01,
1.026835137362170219e-01,
]).reshape((-1, 1))

# Similar to b2 but takes shorter steps (and cheats the reward function by
# veering off to one side).
b4 = np.array([
7.932323594614991702e-01,
2.417523841346104832e-02,
-3.799954860905408460e-01,
1.111652927157003035e-01,
4.505880339777921240e-01,
-3.561543977275546946e-01,
3.848848174979641046e-02,
8.111822670230919852e-02,
]).reshape((-1, 1))

# Best gait that does not sense foot contact.
b5 = np.array([
3.276537604032837003e-01,
-1.434436935541073543e-01,
-5.828843916042253381e-02,
1.035756304579435322e-01,
2.329927247942931157e-01,
-4.275840260264782144e-01,
2.700659007320869304e-02,
2.015317691669641850e-02,
6.731144574614608689e-02,
]).reshape((-1, 1))

#EMBED_B5 = embed_action(b5)

if __name__ == "__main__":
    from darwin_env import DarwinEnv
    env = DarwinEnv()
    env.sdf_loader.ground_width = 8.0
    #b_ckpt = np.loadtxt('data/learn_llc/mean.txt')
    opzer = init_opzer(env, b0)
    learn(opzer, 300)
    embed()
