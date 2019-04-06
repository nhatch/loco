import numpy as np
from IPython import embed
from simbicon_3D import Simbicon3D

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
    if render is not None:
        print("Distance achieved:", dist, "Penalty:", penalty)
    return -i*length + penalty

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
    try:
        for i in range(n_iters):
            val = opzer.f(opzer.mean, render=4)
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
b1 = np.array(
      [[-0.33859163],
       [-0.13842327],
       [ 0.6797675 ],
       [ 0.30470689],
       [ 1.57372148],
       [ 0.22417648],
       [0],
       [0],
       ])

# Most recent trained gait; adjusts balance gains
b2 = np.array([
-1.063469804507329863e-01,
4.722661697983582263e-02,
-4.860981230322012486e-02,
1.301636224517406237e-01,
1.834327175617274375e-03,
-4.033243178337514445e-01,
5.302104274635328213e-02,
1.032243505121583027e+00,
]).reshape((-1, 1))


if __name__ == "__main__":
    from darwin_env import DarwinEnv
    env = DarwinEnv(Simbicon3D)
    env.sdf_loader.ground_width = 8.0
    p = np.zeros(len(controllable_params))
    #test(env, REFERENCE_STEP_LENGTH, p, r=8)
    opzer = init_opzer(env, b0)
    #opzer.f(b2, render=2, terminate_on_slip=False)
    learn(opzer, 40)
    embed()
