import numpy as np
import itertools
import pyflann


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
        print('search point', p_in)

        search_res, _ = self._flann.nn_index(p_in, k)
        print('result indexes', search_res)
        knns = self.__space[search_res]
        print('knns', knns)
        p_out = []
        for p in knns:
            p_out.append(self.export_point(p))

        self.plot_space(additional_points=[p_in])
        return np.array(p_out)

    def import_point(self, point):
        return (point - self._low) / self._range

    def export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def plot_space(self, additional_points=None):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import util.my_plotlib as mplt

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
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for x, y, z in space:

                ax.scatter(x, y, z)

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            plt.show()


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

    max_points = 20
    # s = Space([-1], [1], max_points)
    # s = Space([-1, -2], [1, 2], max_points)
    s = Space([-1, -2, -3], [1, 2, 3], max_points)
    # s = Space([-1, -2, -3, -4], [1, 2, 3, 4], max_points)
    # s = Space([-1, -2, -3, -4, -5], [1, 2, 3, 4, 5], max_points)
    # s = Space([-1, -2, -3, -4, -5, -6], [1, 2, 3, 4, 5, 6], max_points)

    print(s.get_space())
    print(s.shape())

    search_point = np.array([[-0.7, 1.2, -3]])
    # search_point = np.array([[0.15, 0.8, 0]])
    print(s.search_point([-0.7, 1.2, -3], 3))
