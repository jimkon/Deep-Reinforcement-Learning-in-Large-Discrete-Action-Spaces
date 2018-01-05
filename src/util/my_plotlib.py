import numpy as np
import matplotlib.pyplot as plt


class Line:
    def __init__(self, x, y, line_width=1, line_color='black', text='', style='-'):
        self.x = x
        self.y = y
        self.width = line_width
        self.color = line_color
        self.text = text
        self.style = style

    def plot(self, fig=None, k=2):
        if fig is None:
            plt.figure()
            plt.grid(True)
            plt.ylabel("y")
            plt.xlabel("x")

        max_y, min_y = self.y_range()

        plt.plot(self.x, self.y, self.color, linewidth=self.width, linestyle=self.style)
        plt.text(0.05 * len(self.y), k * 0.1 * (max_y - min_y),
                 self.text, color=self.color)

        if fig is None:
            plt.show()

    def y_range(self):
        return np.amin(self.y), np.amax(self.y)


class Function(Line):

    def __init__(self, x, func, line_width=1, line_color='black', text='', style='-'):
        y = [func(i) for i in x]
        super().__init__(x, y, line_width, line_color, text, style)


class Constant(Line):

    def __init__(self, x, c, line_width=1, line_color='black', text='', style='-'):
        x = [x[0], x[len(x) - 1]]
        y = [c] * len(x)
        super().__init__(x, y, line_width, line_color, text, style)


def plot_lines(lines, seps=None, grid_flag=True):
    fig = plt.figure()
    plt.grid(grid_flag)
    plt.ylabel("y")
    plt.xlabel("x")
    max_y = []
    min_y = []
    count = 0
    for line in lines:
        count += 1
        line.plot(fig=fig, k=count)
        temp = line.y_range()
        min_y.append(temp[0])
        max_y.append(temp[1])

    min_y = np.amin(min_y)
    max_y = np.amax(max_y)

    for s in seps:
        plt.plot([s - 0.001, s + 0.001], [min_y, max_y], 'r', linewidth=0.5)

    plt.show()
