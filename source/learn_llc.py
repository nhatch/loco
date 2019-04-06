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
        sp.TORSO_WORLD,
        sp.STANCE_HIP_ROLL_EXTRA,
        sp.STANCE_ANKLE_RELATIVE,
        sp.UP_IDX+sp.SWING_HIP_WORLD,
        sp.UP_IDX+sp.SWING_KNEE_RELATIVE,
        sp.UP_IDX+sp.STANCE_KNEE_RELATIVE,
        sp.POSITION_BALANCE_GAIN,
        sp.VELOCITY_BALANCE_GAIN,
        ])

def test(env, length, param_setting, render=None, n=100, terminate_on_slip=True):
    GL = env.consts().GROUND_LEVEL
    seed = np.random.randint(100000)
    env.reset(seed=seed, random=0.005, video_save_dir=None, render=render)
    env.sdf_loader.put_grounds([[-3.0, GL, 0]])
    c = env.controller
    penalty = 0.0
    for i in range(n):
        stance_com = foot_center(env, c.stance_idx)
        obs, terminated = env.simulate(np.zeros(3), target_heading=0.0, action=param_setting,
                put_dots=True)
        if terminated:
            penalty += 100.0
        penalty += 0.5*(c.stance_heel[1] - GL) # Encourage heelstrikes, not toestrikes
        slip_dist = slipping(env, stance_com)
        if render is not None:
            print("Slip:", slip_dist)
        if i == 0:
            slip_dist /= 2 # First steps are weird, don't penalize as heavily
        penalty += slip_dist
        if slip_dist > MAX_SLIPPING and terminate_on_slip:
            terminated = True
        if terminated:
            break
    penalty += np.abs(c.stance_heel[0] - i*length)
    dist = i*length
    if render is not None:
        print("Distance achieved:", dist, "Penalty:", penalty)
    return -dist + penalty

def foot_center(env, idx):
    c = env.controller
    foot = c.ik.get_bodynode(idx, env.consts().FOOT_BODYNODE_OFFSET)
    return foot.com()

def slipping(env, prev_stance_com):
    c = env.controller
    curr_swing_com = foot_center(env, c.swing_idx)
    slip_distance = np.linalg.norm(prev_stance_com - curr_swing_com)
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
    plt.plot(vals)
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
-1.876361897471146567e-01,
-3.267255715298817975e-01,
-1.224211771795346709e-01,
3.866096294088597896e-01,
-5.428567881104896103e-02,
6.453115104200566332e-01,
-1.543063808195492270e-01,
3.479733144470131267e-01,
]).reshape((-1, 1))


if __name__ == "__main__":
    from darwin_env import DarwinEnv
    env = DarwinEnv(Simbicon3D)
    env.sdf_loader.ground_width = 8.0
    p = np.zeros(len(controllable_params))
    #test(env, REFERENCE_STEP_LENGTH, p, r=8)
    opzer = init_opzer(env, b0)
    #opzer.f(b3, render=2, terminate_on_slip=False)
    #learn(opzer, 40)
    embed()
