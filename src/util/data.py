#!/usr/bin/python3
import json


def load(file_name):
    data = Data()
    with open(file_name, 'r') as f:
        data.merge(json.load(f))
    return data


class Data:

    DATA_TEMPLATE = '''
    {
        "id":0,
        "agent":{
          "name":"default_name",
          "max_actions":0,
          "k":0,
          "version":0
        },
        "experiment":{
          "name":"no_exp",
          "actions_low":null,
          "actions_high":null,
          "number_of_episodes":0
        },
        "simulation":{
          "episodes":[]
        }

    }
    '''

    EPISODE_TEMPLATE = '''
    {
        "states":[],
        "actions":[],
        "actors_actions":[],
        "ndn_actions":[],
        "rewards":[]
    }
    '''

    def __init__(self):
        self.data = json.loads(self.DATA_TEMPLATE)
        self.episode = json.loads(self.EPISODE_TEMPLATE)
        self.temp_saves = 0

    def set_id(self, n):
        self.data['id'] = n

    def set_agent(self, name, max_actions, k, version):
        self.data['agent']['name'] = name
        self.data['agent']['max_actions'] = max_actions
        self.data['agent']['k'] = k
        self.data['agent']['version'] = version

    def set_experiment(self, name, low, high, eps):
        self.data['experiment']['name'] = name
        self.data['experiment']['actions_low'] = low
        self.data['experiment']['actions_high'] = high
        self.data['experiment']['number_of_episodes'] = eps

    def set_state(self, state):
        self.episode['states'].append(state)

    def set_action(self, action):
        self.episode['actions'].append(action)

    def set_actors_action(self, action):
        self.episode['actors_actions'].append(action)

    def set_ndn_action(self, action):
        self.episode['ndn_actions'].append(action)

    def set_reward(self, reward):
        self.episode['rewards'].append(reward)

    def end_of_episode(self):
        self.data['simulation']['episodes'].append(self.episode)
        self.episode = json.loads(self.EPISODE_TEMPLATE)

    def get_file_name(self):
        return 'data_{}_{}_{}#{}'.format(self.get_episodes(),
                                         self.get_agent_name(),
                                         self.get_experiment(),
                                         self.get_id())

    def get_episodes(self):
        return self.data['experiment']['number_of_episodes']

    def get_agent_name(self):
        return '{}{}'.format(self.data['agent']['name'],
                             self.data['agent']['version'])

    def get_id(self):
        return self.data['id']

    def get_experiment(self):
        return self.data['experiment']['name']

    def print_data(self):
        print(json.dumps(self.data, indent=2))

    def merge(self, data_in):
        if type(data_in) is Data:
            data = data_in.data
        else:
            data = data_in

        self.set_id(data['id'])

        self.set_agent(data['agent']['name'],
                       data['agent']['max_actions'],
                       data['agent']['k'],
                       data['agent']['version'])
        self.set_experiment(data['experiment']['name'],
                            data['experiment']['actions_low'],
                            data['experiment']['actions_high'],
                            data['experiment']['number_of_episodes'])

        for ep in data['simulation']['episodes']:
            self.set_state(ep['states'])
            self.set_action(ep['actions'])
            self.set_actors_action(ep['actors_actions'])
            self.set_ndn_action(ep['ndn_actions'])
            self.set_reward(ep['rewards'])
            self.end_of_episode()

    def save(self, path='', final_save=True):
        with open(path + self.get_file_name() + '.json', 'w') as f:
            json.dump(self.data, f, indent=2)

    def temp_save(self):
        self.save(path='temp/' + str(self.temp_saves), final_save=False)
        self.temp_saves += 1
        self.data = json.loads(self.DATA_TEMPLATE)


if __name__ == '__main__':

    import numpy as np

    # d = load('data_10000_agent_name4_exp_name#0.json')
    d = Data()
    d.set_agent('agent_name', 1000, 10, 4)
    d.set_experiment('exp_name', [-2, -3], [3, 2], 10000)
    #
    d.print_data()

    for i in range(10):
        d.set_state([i, i, i, i])
        d.set_action([i, i])
        d.set_actors_action([i, i])
        d.set_ndn_action([i, i])
        d.set_reward([i, i])
        if i % 3 == 0:
            d.end_of_episode()

    d.print_data()
    d.save()
