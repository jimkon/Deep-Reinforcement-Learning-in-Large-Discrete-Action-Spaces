import numpy as np
from my_plotlib import *
from data_old import *
import math
import data_graph
import gym
from gym.spaces import Box, Discrete

"""data wrapper for agent fields, and some graphs"""


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


def plot_average_reward(fd):
    rewards = fd.get_data('rewards')
    batch_size = max(1, int(len(rewards) / 100))
    batches = data_graph.break_into_batches(rewards, batch_size)
    adaption_episode = fd.get_adaption_episode()

    total = 0
    avg = []
    count = 0

    total_adaption_ignored = 0
    avg_adaption_ignored = []
    count_adaption_ignored = 0

    for batch in batches:
        batch_sum = np.sum(batch)
        total += batch_sum
        count += len(batch)
        avg.append(total / count)
        if adaption_episode > 0:
            if count > adaption_episode:
                total_adaption_ignored += batch_sum
                count_adaption_ignored += batch_size
                avg_adaption_ignored.append(total_adaption_ignored / count_adaption_ignored)

    x = np.arange(len(avg)) * batch_size
    lines = [Line(x, avg, text='avg=' + str(avg[len(avg) - 1]), line_color='m')]

    if adaption_episode > 0:
        x_adt = adaption_episode + np.arange(len(avg_adaption_ignored)) * batch_size
        lines.append(Line(x_adt, avg_adaption_ignored,
                          text='avg(ignore adaption)=' + str(avg_adaption_ignored[len(avg_adaption_ignored) - 1]), line_color='r'))

    # fit in x axis
    # adaption_episode = int(round(adaption_episode / batch_size) * batch_size)
    lines.append(Line([adaption_episode, adaption_episode],
                      [0, avg_adaption_ignored[0]],
                      line_color='b--',
                      text='adaption time={} steps({} episodes)'.format(
                          fd.get_adaption_time(), adaption_episode)))

    plot_lines(lines)


def plot_actions(fd, episodes=None, action_space_flag=False):
    lines = []

    data = []

    seps = []
    if episodes is None:
        data = fd.get_data('actions')
    else:
        for ep in episodes:
            data.extend(fd.get_episode_data('actions', ep))
            seps.append(len(data) - 0.5)

    if len(seps) == 1 or len(seps) > 1000:
        seps = []
    x = np.arange(len(data))
    if action_space_flag:
        action_space = fd.get_data('action_space')
        lines.extend((Constant(x, k, line_color='#a0a0a0') for k in action_space))

    try:
        if episodes is None:
            cont_actions = fd.get_data('actors_result')
        else:
            cont_actions = []
            for ep in episodes:
                cont_actions.extend(fd.get_episode_data('actors_result', ep))
        lines.append(Line(x, cont_actions, line_color='#000000', text='actors action'))
    except Exception as e:
        pass

    lines.append(Line(x, data, line_color='-o', line_width=0.5, text='discrete action'))
    plot_lines(lines, seps, grid_flag=not action_space_flag,
               axis_labels={'y': 'action space', 'x': 'steps'})


def plot_actions_error(fd):
    lines = []

    actions = fd.get_data('actions')
    cont_actions = fd.get_data('actors_result')
    result = np.abs(actions - cont_actions)

    batch_size = max(1, int(len(result) / 50))
    batches = data_graph.break_into_batches(result, batch_size)
    avg = []
    total = 0
    count = 0
    for batch in batches:
        total = np.sum(batch)
        count = len(batch)
        avg.append(total / count)
    x = np.arange(len(avg)) * batch_size
    lines.append(Line(x, avg, text='average', line_color='b'))

    adt = fd.get_adaption_time()
    if adt > 0:
        adt = int(round(adt / batch_size) * batch_size)
        lines.append(Line(adt,
                          avg[int(round(adt / batch_size))],
                          line_color='o',
                          text='adaption time'))

    plot_lines(lines)


def plot_action_distribution(fd, batches=-1):
    lines = []

    # different lines for each batch of episodes
    number_of_episodes = fd.get_number_of_episodes()
    if batches < 2:
        batches = 2
    batch_size = int(number_of_episodes / batches)
    eps = [int(item) for item in np.linspace(0, number_of_episodes, batches)]
    colors = np.linspace(0xbbbb11, 0x000011, batches - 1)
    action_space_length = len(fd.get_data('action_space'))
    for i in range(batches - 1):
        print('Batch', i, '-', i + 1)
        episodes = np.arange(eps[i], eps[i + 1])
        data = fd.get_episodes_data('actions', episodes)

        min_action = np.amin(data)
        max_action = np.amax(data)

        y, x = np.histogram(data, bins=math.ceil(action_space_length * .1), density=False)
        # y, x = np.histogram(data, bins=6, density=True)

        x = np.linspace(min_action, max_action, len(y))
        non_zero = np.where(y > 0)
        if batches > 2:
            x = x[non_zero]
            y = y[non_zero]

        y = y / np.sum(y)
        lines.append(Line(x, y, style='-',
                          text='{}-{}'.format(eps[i], eps[i + 1]),
                          line_color='#{:06X}'.format(int(colors[i]))))

    plot_lines(lines, axis_labels={'y': 'Pr', 'x': 'action space'})


def plot_actions_statistics(fd):
    lines = []

    data = fd.get_data('actions')
    y, x = np.histogram(data, bins=len(fd.get_data('action_space')), density=False)

    total = np.sum(y)
    y = np.sort(y)[::-1]
    y = y / total

    sum_y = np.zeros(len(y))
    sum_y[0] = y[0]
    for i in range(1, len(y)):
        sum_y[i] = y[i] + sum_y[i - 1]

    sum_probs = [0.8, 0.9, 0.99]
    for p in sum_probs:
        x_p = np.where(sum_y < p)[0]
        if len(x_p) < 1:
            continue
        x_p = x_p[len(x_p) - 1] + 1
        lines.append(Line(x_p, p, line_color='o',
                          text='{} ({} actions)'.format(p, x_p)))

    all_actions = int(fd.get_perc_of_unique_actions_used() * len(fd.get_data('action_space')))
    lines.append(Line(all_actions, 1, line_color='o',
                      text='{} ({} actions)'.format(1, all_actions)))

    x = np.arange(len(y))

    lines.append(Line(x, y, text='f', line_color='b'))
    lines.append(Line(x, sum_y, text='F', line_color='r'))
    plot_lines(lines, axis_labels={'y': 'Pr', 'x': 'actions'})


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

    def add_field_to_old_version(self, field, value):
        self.load()
        self.add_array(field)
        self.add_to_array(field, value)
        self.save()

    def remove_fields(self, fields, save_to=None):
        try:
            self.load()
            for key in fields:
                self.reset_field(key)
            self.save(path=save_to)
        except Exception as e:
            print("Failed to open and modify " + self.name)

    def get_number_of_episodes(self):
        return len(self.get_data('done'))

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

    # searches for the first window with average greater than the threshold
    def get_adaption_episode(self, reward_threshold=10, window=20):
        rewards = self.get_data('rewards')
        total = 0
        if len(rewards) > window:
            total = np.sum(rewards[:window])

        i = 0
        while i < len(rewards) - window and total / window <= reward_threshold:
            total += rewards[i + window - 1] - rewards[i]
            i += 1

        return i

    def get_adaption_time(self, reward_threshold=50):
        first_increase = self.get_adaption_episode(reward_threshold)
        adaption_time = len(self.get_episodes_data('actions', np.arange(first_increase)))
        return adaption_time

    def get_perc_of_unique_actions_used(self, used_more_than=1):
        data = self.get_data('actions')

        unique = np.unique(data, return_counts=True)[1]
        action_space_length = len(self.get_data('action_space'))
        return len(np.where(unique >= used_more_than)[0]) / action_space_length
