
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

KEYS_TO_PLOT = ['total_reward']
SETTINGS = {
        "total_reward": "red",
        "max_error": "red",
        "n_steps": "black",
        }

def plot_results(results, history, save_dir):
    for settings_name in results.keys():
        lines = []
        labels = []
        for k in KEYS_TO_PLOT:
            data = np.array(results[settings_name][k])
            # Index `1` gives total_steps for both RandomSearchBaseline and LearnInverseDynamics.
            # TODO make this interface more robust/obvious.
            x = map(lambda h: h[1], history)
            x = [0] + list(x)
            color = SETTINGS[k]
            mean = np.mean(data, 1)
            line, = plt.plot(x, mean, color=color)
            #plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
            plt.fill_between(x, np.min(data, 1), np.max(data, 1), color=color, alpha=0.2)
            labels.append(k)
            lines.append(line)

        plt.title(settings_name)
        plt.xlabel("Total number of simulated footsteps taken")
        plt.ylabel("Total reward")
        plt.legend(lines, labels)

        sns.set_style('white')
        sns.despine()

        plt.savefig(save_dir + '{}.png'.format(settings_name))
        plt.clf()
