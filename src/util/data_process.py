#!/usr/bin/python3
from data import *
import data_graph
import numpy as np
import matplotlib.pyplot as plt
import math


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
        if start == end:
            continue
        res.append(func(data[start:end]))

    return res


def break_into_batches(data, batches):
    l = len(data)
    batch_size = int(math.ceil(l / (batches)))
    res = []
    for i in range(batches):
        res.append(data[i * batch_size:(i + 1) * batch_size])

    return res


class Data_handler:

    def __init__(self, filename):
        self.data = load(filename)
        self.episodes = self.data.data['simulation']['episodes']

    def get_episode_data(self, field):
        result = []
        for i in self.episodes:
            data = i[field]

            if isinstance(data, list) and isinstance(data[0], list):
                result.extend(data)
            else:
                result.append(data)
        return result

    def get_full_episode_rewards(self):
        rewards = self.get_episode_data('rewards')
        total_rewards = []
        for episode_rewards in rewards:
            total_rewards.append(np.sum(episode_rewards))
        return total_rewards

    def get_adaption_episode(self, reward_threshold=10, window=20):
        rewards = self.get_full_episode_rewards()
        avg = np.array(apply_func_to_window(rewards, 20, np.average))
        adaption = np.where(avg > reward_threshold)[0][0]
        return adaption


# plots

    def plot_rewards(self):

        rewards = self.get_full_episode_rewards()
        # print(rewards)
        episodes = len(rewards)
        batch_size = int(episodes * .01)

        plt.subplot(211)

        total_avg = average_timeline(rewards)
        plt.plot(total_avg, 'm', label='total avg: {}'.format(total_avg[len(total_avg) - 1]))

        avg = apply_func_to_window(rewards, batch_size, np.average)
        plt.plot(avg, 'g', label='batch avg')

        maxima = apply_func_to_window(rewards, batch_size, np.max)
        plt.plot(maxima, 'r', linewidth=1, label='max')

        minima = apply_func_to_window(rewards, batch_size, np.min)
        plt.plot(minima, 'b', linewidth=1, label='min')

        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(True)

        plt.subplot(212)

        hist = plt.hist(rewards, facecolor='g', alpha=0.75,  rwidth=0.8)
        max_values = int(hist[0][len(hist[0]) - 1])
        x_max_values = int(hist[1][len(hist[0]) - 1])
        plt.annotate(str(max_values), xy=(x_max_values, int(max_values * 1.1)))

        plt.ylabel("Distribution")
        plt.xlabel("Value")
        plt.yscale("log")

        plt.grid(True)
        plt.show()

    def plot_average_reward(self):
        rewards = self.get_full_episode_rewards()

        window_size = int(len(rewards) * .05)
        w_avg = apply_func_to_window(rewards, window_size, np.average)
        plt.plot(w_avg, 'g--', label='widnowed avg (w_size {})'.format(window_size))

        avg = average_timeline(rewards)
        plt.plot(avg, label='average: {}'.format(avg[len(avg) - 1]))

        adaption_time = self.get_adaption_episode()
        plt.plot(adaption_time, avg[adaption_time], 'bo',
                 label='adaption time: {}'.format(adaption_time))

        avg_ignore_adaption = np.array(average_timeline(rewards[adaption_time:]))
        plt.plot(np.arange(adaption_time, len(rewards)), avg_ignore_adaption,
                 label='average(ignore adaption): {}'.format(avg_ignore_adaption[len(avg_ignore_adaption) - 1]))

        argmax = np.argmax(avg_ignore_adaption)
        # print(argmax, adaption_time, len(avg_ignore_adaption))
        plt.plot(argmax + adaption_time, avg_ignore_adaption[argmax], 'ro',
                 label='max: {}'.format(avg_ignore_adaption[argmax]))

        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_distribution(self):
        assert len(self.data.data['experiment']['actions_low']
                   ) == 1, 'This function works only for 1-dimensional action space'
        picked_actions = np.array(self.get_episode_data('actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))
        picked_actions = np.reshape(picked_actions, (len(picked_actions)))
        actors_actions = np.reshape(actors_actions, (len(actors_actions)))
        plt.hist([picked_actions, actors_actions], bins=100,
                 label=['{} actions'.format(len(picked_actions)),
                        'continuous actions'])

        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_distribution_over_time(self, number_of_batches=5, bins=30):
        assert len(self.data.data['experiment']['actions_low']
                   ) == 1, 'This function works only for 1-dimensional action space'
        picked_actions = np.array(self.get_episode_data('actions'))
        batches = break_into_batches(picked_actions, number_of_batches)
        res = []
        count = 0
        for batch in batches:
            hist, bins = np.histogram(batch, bins=np.linspace(0, 1, bins))
            count += 1
            plt.plot(bins[1:], hist, linewidth=1, label='t={}%'.format(
                100 * count / number_of_batches))
            # plt.hist(batch, bins=30, histtype='stepfilled', label=str(count))

        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_error(self):
        ndn = np.array(self.get_episode_data('ndn_actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))

        error = np.sqrt(np.sum(np.square(ndn - actors_actions), axis=1))  # square error
        # plt.plot(error, label='error')

        w_avg = apply_func_to_window(error, 1000, np.average)
        plt.plot(w_avg, label='w error')

        avg_error = average_timeline(error)
        plt.plot(avg_error, label='avg_error :{}'.format(avg_error[len(avg_error) - 1]))

        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    dh = Data_handler('results/obj/data_10000_Wolp3_Inv100k10#0.json.zip')
    # dh = Data_handler('results/obj/data_10000_agen4_exp1000k10#0.json.zip')
    # dh = Data_handler('results/obj/data_10000_Wolp3_Inv10000k1000#0.json.zip')
    print("loaded")

    # dh.get_full_episode_rewards()
    # exit()
    # dh.plot_rewards()
    # dh.plot_rewards()
    # dh.plot_average_reward()
    # dh.plot_action_distribution()
    # dh.plot_action_distribution_over_time()
    dh.plot_action_error()
