import numpy as np
from my_plotlib import *
from data import *
import math
import data_graph
import gym
from gym.spaces import Box, Discrete


def get_action_space(env):
    low = 0
    high = 0
    if isinstance(env.action_space, Box):
        low = env.action_space.low[0]
        high = env.action_space.high[0]
    else:
        low = 0
        high = env.action_space.n

    return low, high


def plot_rewards(fd):
    data = fd.get_data('rewards')

    data_graph.plot_data(data, batch_size=-1, file_name='rewards')


def plot_actions(fd, episodes=None, action_space=None):
    lines = []

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
    if action_space is not None:
        lines.extend((Constant(x, k, line_color='#a0a0a0') for k in action_space))

    lines.append(Line(x, data, line_color='-o'))
    plot_lines(lines, seps, grid_flag=action_space is None)


def plot_action_distribution(fd):
    lines = []

    # different lines for each batch of episodes
    number_of_episodes = fd.get_number_of_episodes()
    batches = int(min(5, math.ceil(number_of_episodes / 800)))
    batch_size = int(number_of_episodes / batches)
    eps = [int(item) for item in np.linspace(0, number_of_episodes, batches)]
    colors = np.linspace(0xaa00aa, 0x00ff00, batches + 1)

    for i in range(batches - 1):
        episodes = np.arange(eps[i], eps[i + 1])
        data = fd.get_episodes_data('actions', episodes)

        min_action = np.amin(data)
        max_action = np.amax(data)
        # data_len = len(data)
        y, x = np.histogram(data, bins='auto', density=True)

        x = np.linspace(min_action, max_action, len(y))
        non_zero = np.where(y > 0)
        x = x[non_zero]
        y = y[non_zero]
        y = y / np.sum(y)
        lines.append(Line(x, y, style='-',
                          text='{}-{}'.format(eps[i], eps[i + 1]),
                          line_color='#{:06X}'.format(int(colors[i]))))

    plot_lines(lines, log=False)


def plot_actions_statistics(fd):
    lines = []

    data = fd.get_data('actions')
    y, x = np.histogram(data, bins='auto', density=False)
    print(len(y))
    total = np.sum(y)
    y = np.sort(y)[::-1]
    y = y / total

    sum_y = np.zeros(len(y))
    sum_y[0] = y[0]
    for i in range(1, len(y)):
        sum_y[i] = y[i] + sum_y[i - 1]

    sum_probs = [0.8, 0.9, 0.99, 0.999]
    for p in sum_probs:
        x_p = np.where(sum_y < p)[0]
        x_p = x_p[len(x_p) - 1] + 1
        lines.append(Line(x_p, p, line_color='o',
                          text='{} ({} actions)'.format(p, x_p)))

    x = np.arange(len(y))

    lines.append(Line(x, y, text='f'))
    lines.append(Line(x, sum_y, text='F'))
    plot_lines(lines)


def plot_states(fd, episodes=None):
    lines = []

    data = {'s0': [],
            's1': [],
            's2': [],
            's3': [],
            'actions': []}
    seps = []
    if episodes is None:
        data['s0'] = fd.get_data('state_0')
        data['s1'] = fd.get_data('state_1')
        data['s2'] = fd.get_data('state_2')
        data['s3'] = fd.get_data('state_3')
        data['actions'] = fd.get_data('actions')
    else:
        for ep in episodes:
            data['s0'].extend(fd.get_episode_data('state_0', ep))
            data['s1'].extend(fd.get_episode_data('state_1', ep))
            data['s2'].extend(fd.get_episode_data('state_2', ep))
            data['s3'].extend(fd.get_episode_data('state_3', ep))
            data['actions'].extend(fd.get_episode_data('actions', ep))
            seps.append(len(data) - 0.5)

    if len(seps) == 1:
        seps = []
    x = np.arange(len(data['s0']))

    lines.append(Line(x, data['s0'], line_color='b', text='s0'))
    lines.append(Line(x, data['s1'], line_color='g', text='s1'))
    lines.append(Line(x, data['s2'], line_color='r', text='s2'))
    lines.append(Line(x, data['s3'], line_color='m', text='s3'))
    lines.append(Line(x, data['actions'], line_color='black', text='actions', style=':'))

    plot_lines(lines, seps)


def plot_reward_3d(fd, batch_size_ratio=0.1):
    data = fd.get_data('rewards')
    batch_size = math.ceil(batch_size_ratio * len(data))
    print(len(data), batch_size)
    assert batch_size > 0, "int(batch_size*len(data)) has must be > 0"
    Z = add_y_dimension(data, batch_size)
    X = np.arange(Z.shape[1])
    Y = np.arange(Z.shape[0])
    data_graph.plot_surface(X, Y, Z)


def add_y_dimension(data, batch_size):
    max_value = 1001
    min_value = 0
    value_batch_size = 100
    batches = list(data_graph.break_into_batches(data, batch_size))
    z_shape = (math.ceil((max_value - min_value) / value_batch_size),
               len(batches))
    z = np.ones(shape=z_shape)
    count = 0
    for batch in batches:
        for num in batch:
            z[int(num / value_batch_size)][count] += 1
        count += 1

    return np.log(z)


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

    def get_episodes_data(self, field, eps=None):
        res = []
        if eps is None:
            eps = np.arange(self.get_number_of_episodes)
        for ep in eps:
            res.extend(self.get_episode_data(field, ep))
        return res

    def get_full_episode_data(self, ep):
        start, end = self.find_episode(ep)
        clone = self.get_empty_clone()
        for key in self.get_keys():
            clone.set_data(key, self.get_data(key)[start: end + 1])

        r = self.get_data('rewards')[ep]
        clone.set_data('rewards', np.array([r]))
        return clone

    def get_number_of_episodes(self):
        return len(self.get_data('rewards'))
