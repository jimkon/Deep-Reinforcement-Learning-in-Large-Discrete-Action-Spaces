import random
import numpy as np

from util.timer import *

import gym
from gym.spaces import Box, Discrete

from actor_net import ActorNet
from critic_net import CriticNet
from actor_net_bn import ActorNet_bn
from critic_net_bn import CriticNet_bn
from tensorflow_grad_inverter import grad_inverter

from collections import deque

import pyflann

MAX_ACTION_SPACE_SIZE = 1e6


class Agent:

    def __init__(self, env):
        # checking state space
        if isinstance(env.observation_space, Box):
            self.observation_space_size = env.observation_space.shape[0]
        else:
            self.observation_space_size = env.observation_space.n

        # checking action space
        if isinstance(env.action_space, Box):
            self.action_space_size = env.action_space.shape[0]
            self.continious_action_space = True
            self.low = env.action_space.low
            self.high = env.action_space.high
        else:
            self.action_space_size = env.action_space.n
            self.continious_action_space = False
            self.low = 0
            self.high = env.action_space.n

    def act(self, state):
        pass

    def observe(self, episode):
        pass

    # shaping input states and actions
    def _np_shaping(self, array, is_state):

        number_of_elements = array.shape[0] if len(array.shape) > 1 else 1
        size_of_element = self.observation_space_size if is_state else self.action_space_size

        res = np.array(array)
        res.shape = (number_of_elements, size_of_element)
        return res

    def get_name(self):
        return 'Agent'


class RandomAgent(Agent):

    def act(self, state):
        if self.continious_action_space:
            res = self.low + (self.high - self.low) * np.random.uniform(size=len(self.low))
            return res
        else:
            return random.randint(self.low, self.high - 1)

    def get_name(self):
        return 'Random' + super().get_name()


class DiscreteRandomAgent(RandomAgent):

    def __init__(self, env, max_actions=10):
        super().__init__(env)
        if self.continious_action_space:
            self.discrete_actions = np.linspace(self.low, self.high, max_actions)
        else:
            self.discrete_actions = np.arange(self.low, self.high)
        self.discrete_actions = list(self.discrete_actions)

    def act(self, state):
        return random.sample(self.discrete_actions, 1)[0]

    def get_name(self):
        return 'Discrete' + super().get_name()


class DDPGAgent(Agent):
    ''' stevenpjg's implementation of DDPG algorithm '''

    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.99

    def __init__(self, env, is_batch_norm=False, is_grad_inverter=True):
        super().__init__(env)
        if is_batch_norm:
            self.critic_net = CriticNet_bn(self.observation_space_size,
                                           self.action_space_size)
            self.actor_net = ActorNet_bn(self.observation_space_size,
                                         self.action_space_size)

        else:
            self.critic_net = CriticNet(self.observation_space_size,
                                        self.action_space_size)
            self.actor_net = ActorNet(self.observation_space_size,
                                      self.action_space_size)

        self.is_grad_inverter = is_grad_inverter
        self.replay_memory = deque()

        self.time_step = 0

        action_max = np.array(env.action_space.high).tolist()
        action_min = np.array(env.action_space.low).tolist()
        action_bounds = [action_max, action_min]
        self.grad_inv = grad_inverter(action_bounds)

        self.train_timings = Time_stats("Training",
                                        ['p_t', 'q_t', 'y',
                                         'train_q', 'train_p',
                                         'up_q_t', 'up_p_t'])

    def get_name(self):
        return 'DDPG' + super().get_name()

    def act(self, state):
        state = self._np_shaping(state, True)
        return self.actor_net.evaluate_actor(state).astype(float)

    def observe(self, episode):
        episode['obs'] = self._np_shaping(episode['obs'], True)
        episode['action'] = self._np_shaping(episode['action'], False)
        episode['obs2'] = self._np_shaping(episode['obs2'], True)
        self.add_experience(episode)

    def add_experience(self, episode):
        self.replay_memory.append(episode)

        self.time_step += 1
        if len(self.replay_memory) > type(self).REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

        if len(self.replay_memory) > type(self).BATCH_SIZE:
            res = self.train()
            return res
        else:
            return None

    def minibatches(self):
        batch = random.sample(self.replay_memory, type(self).BATCH_SIZE)
        # state t
        state = self._np_shaping(np.array([item['obs'] for item in batch]), True)
        # action
        action = self._np_shaping(np.array([item['action'] for item in batch]), False)
        # reward
        reward = np.array([item['reward'] for item in batch])
        # state t+1
        state_2 = self._np_shaping(np.array([item['obs2'] for item in batch]), True)
        # doneA
        done = np.array([item['done'] for item in batch])

        return state, action, reward, state_2, done

    def train(self):
        # sample a random minibatch of N transitions from R
        state, action, reward, state_2, done = self.minibatches()

        actual_batch_size = len(state)

        self.train_timings.reset_timers()
        target_action = self.actor_net.evaluate_target_actor(state)
        self.train_timings.add_time('p_t')  # ------
        # Q'(s_i+1,a_i+1)
        q_t = self.critic_net.evaluate_target_critic(state_2, target_action)
        self.train_timings.add_time('q_t')  # ------

        y = []  # fix initialization of y
        for i in range(0, actual_batch_size):

            if done[i]:
                y.append(reward[i])
            else:
                y.append(reward[i] + type(self).GAMMA * q_t[i][0])  # q_t+1 instead of q_t

        y = np.reshape(np.array(y), [len(y), 1])
        self.train_timings.add_time('y')  # ------

        # Update critic by minimizing the loss
        self.critic_net.train_critic(state, action, y)
        self.train_timings.add_time('train_q')  # ------
        # Update actor proportional to the gradients:
        # action_for_delQ = self.act(state)  # was self.evaluate_actor instead of self.act
        action_for_delQ = self.actor_net.evaluate_actor(state)  # dont need wolp action

        if self.is_grad_inverter:
            del_Q_a = self.critic_net.compute_delQ_a(state, action_for_delQ)  # /BATCH_SIZE
            del_Q_a = self.grad_inv.invert(del_Q_a, action_for_delQ)
        else:
            del_Q_a = self.critic_net.compute_delQ_a(state, action_for_delQ)[0]  # /BATCH_SIZE

        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(state, del_Q_a)
        self.train_timings.add_time('train_p')  # ------

        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.train_timings.add_time('up_q_t')  # ------
        self.actor_net.update_target_actor()
        self.train_timings.add_time('up_p_t')  # ------
        self.train_timings.increase_count(n=actual_batch_size / self.BATCH_SIZE)

    def get_train_timings(self):
        return self.train_timings

    def load_expierience(self, file_path='results/last_episode'):
        from numpy import array

        dicts_from_file = {}
        with open(file_path, 'r') as inf:
            dict_from_file = eval(inf.read())
        print(dicts_from_file)


class WolpertingerAgent(DDPGAgent):

    def __init__(self, env, max_actions=1e6, k_nearest_neighbors=10):
        super().__init__(env)
        if self.continious_action_space:
            self.actions = np.linspace(self.low, self.high, max_actions)
        else:
            self.actions = np.arange(self.low, self.high)
        # self.actions = list(self.actions)
        self.k_nearest_neighbors = k_nearest_neighbors
        print('wolpertinger agent init')
        print('max actions = ', max_actions)
        print('k nearest neighbors =', k_nearest_neighbors)
        # init flann
        self.actions.shape = (len(self.actions), self.action_space_size)
        self.flann = pyflann.FLANN()
        params = self.flann.build_index(self.actions, algorithm='kdtree')
        print('flann init with params->', params)

    def get_name(self):
        return 'Wolp_k=1_' + super().get_name()

    def act(self, state):
        proto_action = super().act(state)
        if self.k_nearest_neighbors <= 1:
            return proto_action

        if len(proto_action) > 1:
            # print('\n\n\nproto action shape', proto_action.shape)
            res = np.array([])
            for i in range(len(proto_action)):
                res = np.append(res, self.wolp_action(state[i], proto_action[i]))
            res.shape = (len(res), 1)
            # print(res)
            # print(res.shape)
            return res
        else:
            return self.wolp_action(state, proto_action)

    def wolp_action(self, state, proto_action):
        actions = self.nearest_neighbors(proto_action)[0]
        # print('--\nproto action', proto_action, 'state', state)
        # print('actions', actions.shape)
        states = np.tile(state, [len(actions), 1])
        # print('states', states.shape)
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        # print('action evalueations', actions_evaluation.shape)
        max_index = np.argmax(actions_evaluation)
        max = actions_evaluation[max_index]
        # print('max', max, '->', max_index)
        return max

    def nearest_neighbors(self, proto_action):
        results, dists = self.flann.nn_index(
            proto_action, self.k_nearest_neighbors)  # checks=params["checks"]
        return self.actions[results]
