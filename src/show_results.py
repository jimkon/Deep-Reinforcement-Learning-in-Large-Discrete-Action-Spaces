import numpy as np
from util.my_plotlib import *
from util.data import *
from util.agent_data import *
import util.data_graph

import gym
from wolp_agent import *


def main():
    # for i in [2511, 2512, 5001, 5002]:
    if True:
        folder = '/'
        episodes = 10000
        actions = 10
        k = 10
        experiment = 'InvertedPendulum-v1'
        v = 3

        name = '{}data_{}_Wolp{}_{}k{}_{}'.format(folder,
                                                  episodes,
                                                  v,
                                                  actions,
                                                  k,
                                                  experiment
                                                  )

        fd = Agent_data(name=name)

        fd.load()
        # fd.print_times(other_keys=fd.get_keys('run'))
        # fd.print_times(other_keys=fd.get_keys('agent_'), total_time_field='count')
        plot_rewards(fd)
        plot_average_reward(fd)
        plot_action_distribution(fd, batches=int(episodes * .1) + 1)

        plot_actions_statistics(fd)


if __name__ == '__main__':
    main()
