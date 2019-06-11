from IPython import embed
from experiment import Experiment
from inverse_dynamics import LearnInverseDynamics
from rs_baseline import RandomSearchBaseline
from cma_baseline import CMABaseline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

SETTINGS = {
        "cim_final_easy": ["green", "CIM", LearnInverseDynamics, [5,6,7,8], 'SETTINGS_2D_EASY'],
        "cim_final": ["green", "CIM", LearnInverseDynamics, [17,18,19,20], 'SETTINGS_2D_HARD'],
        "test_cim_final": ["magenta", "Test CIM", LearnInverseDynamics, [12,13,14,15], 'SETTINGS_2D_HARD'],
        "cma_baseline": ["black", "CMA individual step", CMABaseline, [0], 'SETTINGS_2D_EASY'],
        "cim_3D": ["green", "CIM", LearnInverseDynamics, [1], 'SETTINGS_3D_HARD'],
        "3D_test": ["green", "CIM", LearnInverseDynamics, [1], 'SETTINGS_3D_HARD'],
        "test_cim_3D": ["green", "CIM", LearnInverseDynamics, [1], 'SETTINGS_3D_HARD'],
        "rs_final": ["purple", "ARS baseline", RandomSearchBaseline, [13,14,15,16], 'SETTINGS_2D_EASY'],
        "nocur_final": ["blue", "No curriculum", LearnInverseDynamics, [9,10,11,12], 'SETTINGS_2D_HARD'],
        }

def fill_plot(x, data, color):
    line, = plt.plot(x, np.percentile(data, 50, 1), color=color)
    #plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    plt.fill_between(x, np.percentile(data, 10, 1), np.percentile(data, 90, 1),
            color=color, alpha=0.2)
    return line

def steps_from_history(history):
    # Index `1` gives total_steps for both RandomSearchBaseline and LearnInverseDynamics.
    # TODO make this interface more robust/obvious.
    steps = map(lambda h: h[1], history)
    steps = [0] + list(steps)
    return steps

def load_exp(exp_name, seed):
    name = "{}_{}".format(exp_name, seed)
    eval_settings = SETTINGS[exp_name][4]
    ex = Experiment(None, SETTINGS[exp_name][2], name, [eval_settings])
    return steps_from_history(ex.learn.history), ex.results[eval_settings]['total_reward']

def multiseed_plot(exp_name, seeds=None):
    xx = []
    yy = []
    if seeds is None:
        seeds = SETTINGS[exp_name][3]
    for seed in seeds:
        x,y = load_exp(exp_name, seed)
        xx.append(x)
        yy.append(y)
    xx = np.array(xx)
    final_x = np.mean(xx, 0)
    final_y = np.concatenate(yy, 1)
    color = SETTINGS[exp_name][0]
    return fill_plot(final_x, final_y, color)

def save_plot(filename):
    plt.xlabel("Total number of simulated footsteps taken")
    plt.ylabel("Total reward")

    sns.set_style('white')
    sns.despine()

    plt.savefig(filename)
    plt.clf()

def gen_figures():
    lines = []
    labels = []
    for exp in ['cim_final_easy', 'rs_final']:
        lines.append(multiseed_plot(exp))
        labels.append(SETTINGS[exp][1])
    plt.legend(lines, labels)
    save_plot('../paper/figures/ars_baseline.pdf')

    lines = []
    labels = []
    for exp in ['cim_final', 'nocur_final', 'test_cim_final']:
        lines.append(multiseed_plot(exp))
        labels.append(SETTINGS[exp][1])
    plt.legend(lines, labels, loc='lower right')
    save_plot('../paper/figures/nocur_baseline.pdf')

def plot_single_seed(exp_name, seed):
    multiseed_plot(exp_name, [seed])
    eval_settings = SETTINGS[exp_name][4]
    save_plot('data/{}_{}/{}.pdf'.format(exp_name, seed, eval_settings))

def all_single_seeds():
    for exp_name in SETTINGS.keys():
        seeds = SETTINGS[exp_name][3]
        for seed in seeds:
            plot_single_seed(exp_name, seed)

if __name__ == '__main__':
    #gen_figures()
    plot_single_seed('test_cim_3D', 6)
    plot_single_seed('test_cim_final', 'test')
    #all_single_seeds()
