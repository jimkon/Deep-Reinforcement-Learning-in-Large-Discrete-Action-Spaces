#!/usr/bin/python3
from data import *
import data_graph
import numpy as np
import matplotlib.pyplot as plt


def average_timeline(x):
    res = []
    count = 0
    total = 0
    for i in x:
        total += i
        count += 1
        res.append(total / count)
    return res


def apply_func_to_window(data, window_size, func):
    data_lenght = len(data)
    res = []
    for i in range(data_lenght):
        start = int(max(i - window_size / 2, 0))
        end = int(min(i + window_size / 2, data_lenght - 1))
        res.append(func(data[start:end]))

    return res


class Data_handler:

    def __init__(self, filename):
        self.data = load(filename)
        self.episodes = self.data.data['simulation']['episodes']

    def get_episode_data(self, field):
        result = []
        for i in self.episodes:
            result.extend(i[field])
        return result

    def get_adaption_episode(self, reward_threshold=10, window=20):
        rewards = np.array(self.get_episode_data('rewards'))
        avg = np.array(apply_func_to_window(rewards, 20, np.average))
        adaption = np.where(avg > reward_threshold)[0][0]
        return adaption


# plot functions

    def plot_rewards(self):
        rewards = np.array(self.get_episode_data('rewards'))
        data_graph.plot_data(rewards, batch_size=-1, file_name='rewards')

    def plot_average_reward(self):
        rewards = np.array(self.get_episode_data('rewards'))
        # avg = apply_func_to_window(rewards, 500, np.average)
        avg = average_timeline(rewards)
        plt.plot(avg, label='average: {}'.format(avg[len(avg) - 1]))

        adaption_time = self.get_adaption_episode()
        plt.plot(adaption_time, avg[adaption_time], 'bo',
                 label='adaption time: {}'.format(adaption_time))

        avg_ignore_adaption = average_timeline(rewards[adaption_time:])
        plt.plot(np.arange(adaption_time, len(rewards)), avg_ignore_adaption,
                 label='average(ignore adaption): {}'.format(avg_ignore_adaption[len(avg_ignore_adaption) - 1]))

        argmax = np.argmax(avg_ignore_adaption)
        print(argmax, adaption_time, avg_ignore_adaption)
        plt.plot(argmax + adaption_time, avg_ignore_adaption[argmax],
                 label='max:'.format(avg_ignore_adaption[argmax]))

        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # dh = Data_handler('results/obj/data_10000_agent_name3_exp1000k10#0.json.zip')
    dh = Data_handler('results/obj/data_10000_Wolp3_Inv10k10#0.json.zip')
    # print(dh.get_episode_data('rewards'))
    # dh.get_adaption_episode()
    dh.plot_average_reward()
