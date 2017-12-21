import numpy as np
import pyflann

from ddpg import agent


class WolpertingerAgent(agent.DDPGAgent):

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
        return 'Wolp_v1_k' + str(self.k_nearest_neighbors) + '_' + super().get_name()

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
