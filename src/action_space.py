import numpy as np
import itertools
import pyflann

import util.my_plotlib as mplt
from util.data_graph import plot_3d_points


class Space:

    def __init__(self, low, high, points):
        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)
        self.__space = init_uniform_space([0] * self._dimensions,
                                          [1] * self._dimensions,
                                          points)
        self._flann = pyflann.FLANN()
        self.rebuild_flann()

    def rebuild_flann(self):
        self._index = self._flann.build_index(self.__space, algorithm='kdtree')

    def search_point(self, point, k):
        p_in = self.import_point(point)
        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self.__space[search_res]
        p_out = []
        for p in knns:
            p_out.append(self.export_point(p))

        return np.array(p_out)

    def import_point(self, point):
        return (point - self._low) / self._range

    def export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]

    def plot_space(self, additional_points=None):

        dims = self._dimensions

        if dims > 3:
            print(
                'Cannot plot a {}-dimensional space. Max 3 dimensions'.format(dims))
            return

        space = self.get_space()
        if additional_points is not None:
            for i in additional_points:
                space = np.append(space, additional_points, axis=0)

        if dims == 1:
            lines = []
            for x in space:
                lines.append(mplt.Line([x], [0], line_color='o'))

            mplt.plot_lines(lines)
        elif dims == 2:
            lines = []
            for x, y in space:
                lines.append(mplt.Line([x], [y], line_color='o'))

            mplt.plot_lines(lines)
        else:
            plot_3d_points(space)


def init_uniform_space(low, high, points):
    dims = len(low)
    points_in_each_axis = round(points**(1 / dims))

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))

    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    return np.array(space)


if __name__ == '__main__':

    max_points = 36
    # s = Space([-1], [1], max_points)
    s = Space([-1, -2], [1, 2], max_points)
    # s = Space([-1, -2, -3], [1, 2, 3], max_points)
    # s = Space([-1, -2, -3, -4], [1, 2, 3, 4], max_points)
    # s = Space([-1, -2, -3, -4, -5], [1, 2, 3, 4, 5], max_points)
    # s = Space([-1, -2, -3, -4, -5, -6], [1, 2, 3, 4, 5, 6], max_points)

    # print(s.get_space())
    # print(s.shape())

    search_point = np.array([[-0.7, 1.2]])
    print(s.search_point(search_point, 10))
