import numpy as np
import matplotlib.pyplot as plt


class Line:
    def __init__(self, x, y, line_width=1, line_color='black', text=''):
        self.x = x
        self.y = y
        self.width = line_width
        self.color = line_color
        self.text = text

    def plot(self, fig=None, k=2):
        if fig is None:
            plt.figure()
            plt.grid(True)
            plt.ylabel("y")
            plt.xlabel("x")

        line = self
        max_y = np.amax(line.y)
        min_y = np.amin(line.y)

        plt.plot(line.x, line.y, line.color, linewidth=line.width)
        plt.text(0.05 * len(line.y), k * 0.1 * (max_y - min_y),
                 line.text, color=line.color)

        if fig is None:
            plt.show()


class Function(Line):

    def __init__(self, x, func, line_width=1, line_color='black', text=''):
        y = [func(i) for i in x]
        super().__init__(x, y, line_width, line_color, text)


def plot_lines(lines):
    fig = plt.figure()
    plt.grid(True)
    plt.ylabel("y")
    plt.xlabel("x")
    count = 0
    for line in lines:
        count += 1
        line.plot(fig=fig, k=count)
    plt.show()
