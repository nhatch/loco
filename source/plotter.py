from IPython import embed
from experiment import Experiment
from inverse_dynamics import LearnInverseDynamics
from rs_baseline import RandomSearchBaseline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

KEYS_TO_PLOT = ['total_reward']
SETTINGS = {
        "cim_final_easy": ["green", "CIM", LearnInverseDynamics, [5,6,7,8]],
        "cim_final": ["green", "CIM", LearnInverseDynamics, [17,18,19,20]],
        "rs_final": ["purple", "ARS baseline", RandomSearchBaseline, [13,14,15,16]],
        "nocur_final": ["blue", "No curriculum", LearnInverseDynamics, [9,10,11,12]],
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
    settings_name = 'SETTINGS_2D_EASY' if exp_name in ['cim_final_easy', 'rs_final'] else 'SETTINGS_2D_HARD'
    ex = Experiment(None, SETTINGS[exp_name][2], name, [settings_name])
    return steps_from_history(ex.learn.history), ex.results[settings_name]['total_reward']

def multiseed_plot(exp_name):
    xx = []
    yy = []
    for seed in SETTINGS[exp_name][3]:
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
    for exp in ['cim_final', 'nocur_final']:
        lines.append(multiseed_plot(exp))
        labels.append(SETTINGS[exp][1])
    plt.legend(lines, labels, loc='lower right')
    save_plot('../paper/figures/nocur_baseline.pdf')

if __name__ == '__main__':
    gen_figures()