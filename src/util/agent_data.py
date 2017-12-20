import numpy as np
from my_plotlib import *
from data import *
import data_graph


def main():
    n = 2511
    fd = Agent_data(
        name='data_Wolp_betaDDPGAgent' + str(n))

    fd.load()

    # fd.print_data()

    # fd = fd.get_full_episode_data(1)
    # fd.print_data()

    print_actions(fd, [0, 1])


def print_rewards(fd):
    data = fd.get_data('rewards')

    data_graph.plot_data(data, batch_size=-1, file_name='rewards')


def print_actions(fd, episodes=None):
    data = []
    seps = []
    if episodes is None:
        data = fd.get_data('actions')
    else:
        for ep in episodes:
            data.extend(fd.get_episode_data('actions', ep))
            seps.append(len(data) - 0.5)

    if len(seps) == 1:
        seps = []
    x = np.arange(len(data))
    line = Line(x, data)
    plot_lines([line], seps)


class Agent_data(Data):

    def get_episodes_with_reward_greater_than(self, th):
        return np.where(self.get_data('rewards') >= th)[0]

    def find_episode(self, ep):
        done = self.get_data('done')
        eps = np.where(done == 1)[0]
        return eps[ep - 1] + 1 if ep > 0 else 0, eps[min(ep, len(done))]

    def get_episode_data(self, field, ep):
        s, e = self.find_episode(ep)
        data = self.get_data(field)
        if field == 'rewards':
            return data[ep]
        else:
            return data[s: e + 1]

    def get_full_episode_data(self, ep):
        start, end = self.find_episode(ep)
        clone = self.get_empty_clone()
        for key in self.get_keys():
            clone.set_data(key, self.get_data(key)[start: end + 1])

        r = self.get_data('rewards')[ep]
        clone.set_data('rewards', np.array([r]))
        return clone


if __name__ == '__main__':
    main()
