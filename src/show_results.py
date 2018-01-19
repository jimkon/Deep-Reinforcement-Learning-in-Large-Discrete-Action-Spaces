import numpy as np
from util.my_plotlib import *
from util.data import *
from util.agent_data import *
import util.data_graph


def show():
    folder = 'saved/'
    episodes = 10000
    actions = 1000
    k = 100
    experiment = 'InvertedPendulum-v1'
    v = 3

    name = '{}data_{}_Wolp{}_{}k{}_{}_shrinked'.format(folder,
                                                       episodes,
                                                       v,
                                                       actions,
                                                       k,
                                                       experiment
                                                       )

    fd = Agent_data(name)

    fd.load()

    plot_rewards(fd)

    plot_average_reward(fd)

    print('Printing action distribution... This might take a while. # of actions:',
          len(fd.get_data('actions')))
    batches = min(11, int(fd.get_number_of_episodes() * .001) + 1)
    plot_action_distribution(fd, batches=batches)

    plot_actions_statistics(fd)


if __name__ == '__main__':
    show()
