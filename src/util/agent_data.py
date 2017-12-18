import numpy as np
from my_plotlib import *
from data import *
import data_graph


def main():
    n = 2511
    fd = Agent_data(
        name='data_Wolp_betaDDPGAgent' + str(n))

    fd.load()

    print(fd.find_episode(10))
    print_rewards(fd)
    print_actions(fd, 0)


def print_rewards(fd):
    data = fd.get_data('rewards')

    data_graph.plot_data(data, batch_size=-1, file_name='rewards')


def print_actions(fd, episode):
    data = fd.get_data('actions')
    x = np.arange(len(data))
    line = Line(x, data)
    plot_lines([line])


if __name__ == '__main__':
    main()
