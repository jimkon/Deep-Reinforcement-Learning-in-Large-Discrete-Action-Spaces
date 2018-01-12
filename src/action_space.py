import numpy as np


class Action_space:

    def __init__(self, low, high, max_actions):
        dims = len(low)

        k = round(max_actions**(1 / dims))
        print(k)
        pass


if __name__ == '__main__':
    Action_space([-1, -2, -3], [1, 2, 3], 100)
