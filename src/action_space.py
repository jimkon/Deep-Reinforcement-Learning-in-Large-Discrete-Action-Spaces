import numpy as np
import itertools


class Action_space:

    def __init__(self, low, high, max_actions):
        dims = len(low)

        k = round(max_actions**(1 / dims))
        # print(k)

        axis = []
        for i in range(dims):
            axis.append(list(np.linspace(low[i], high[i], k)))

        # print(axis)

        self.space = []
        for _ in itertools.product(*axis):
            self.space.append(list(_))

        self.space = np.array(self.space)

        # print(space)
        print(self.space.shape)

    def get_space(self):
        return self.space

    def plot_space(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import util.my_plotlib as mplt

        dims = self.space.shape[1]

        if dims > 3:
            print(
                'Cannot plot a {}-dimensional space. Max 3 dimensions'.format(self.space.shape[1]))
            return

        space = a_s.get_space()
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


if __name__ == '__main__':

    max_points = 100
    # a_s = Action_space([-1], [1], max_points)
    # a_s = Action_space([-1, -2], [1, 2], max_points)
    a_s = Action_space([-1, -2, -3], [1, 2, 3], max_points)
    # a_s = Action_space([-1, -2, -3, -4], [1, 2, 3, 4], max_points)
    # a_s = Action_space([-1, -2, -3, -4, -5], [1, 2, 3, 4, 5], max_points)
    # a_s = Action_space([-1, -2, -3, -4, -5, -6], [1, 2, 3, 4, 5, 6], max_points)
    a_s.plot_space()
