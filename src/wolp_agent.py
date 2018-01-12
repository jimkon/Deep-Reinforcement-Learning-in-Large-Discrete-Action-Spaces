import numpy as np
import pyflann
from gym.spaces import Box
from ddpg import agent


class WolpertingerAgent(agent.DDPGAgent):

    def __init__(self, env, max_actions=1e6, k_ratio=0.1):
        super().__init__(env)
        self.experiment = env.spec.id
        if self.continious_action_space:
            self.actions = self.quantize_action_space(self.low, self.high, max_actions)
        else:
            self.actions = np.arange(self.low, self.high)

        self.k_nearest_neighbors = int(max_actions * k_ratio)

        # init flann
        self.actions.shape = (len(self.actions), self.action_space_size)
        self.flann = pyflann.FLANN()
        params = self.flann.build_index(self.actions, algorithm='kdtree')

    def get_name(self):
        return 'Wolp2_{}k{}_{}'.format(len(self.actions), self.k_nearest_neighbors, self.experiment)

    def quantize_action_space(self, low, high, max_actions):
        return np.linspace(low, high, max_actions)

    def get_action_space(self):
        return self.actions

    def act(self, state):
        proto_action = super().act(state)
        if self.k_nearest_neighbors <= 1:
            return proto_action

        if len(proto_action) > 1:
            return 0
            res = np.array([])
            for i in range(len(proto_action)):
                res = np.append(res, self.wolp_action(state[i], proto_action[i]))
            res.shape = (len(res), 1)
            return res
        else:
            return self.wolp_action(state, proto_action)

    def wolp_action(self, state, proto_action):
        debug = False
        actions = self.nearest_neighbors(proto_action)[0]
        if debug:
            print('--\nproto action', proto_action, 'state', state)
        states = np.tile(state, [len(actions), 1])
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        if debug:
            print('action evalueations', actions_evaluation.shape)
        if debug:
            for i in range(len(actions)):
                print(actions[i], 'v', actions_evaluation[i])

        max_index = np.argmax(actions_evaluation)
        max = actions_evaluation[max_index]
        if debug:
            print('max', max, '->', max_index)
        if debug:
            print('result action', actions[max_index])
        # if debug:
        #     exit()
        return actions[max_index]

    def nearest_neighbors(self, proto_action):
        results, dists = self.flann.nn_index(
            proto_action, self.k_nearest_neighbors)  # checks=params["checks"]
        return self.actions[results]
