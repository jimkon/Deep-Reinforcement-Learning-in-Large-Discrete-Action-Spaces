import numpy as np
import matplotlib.pyplot as plt
import math
import ntpath

BATCH_RATIO = 0.01
EXTENSIONS = ['.txt']
DIRECTORY = "/../../results"


def plot_file(file_name):
    data = np.loadtxt(file_name)
    plot_data(data, file_name=file_name)


def plot_data(data, batch_size=-1, file_name="data"):

    data_size = data.shape[0]
    if batch_size == -1:
        batch_size = max(int(data_size * BATCH_RATIO), 1)
    if BATCH_RATIO == 1:
        batch_size = 1
    batches = math.ceil(data_size / batch_size)

    final_data = np.zeros((batches, 4))

    for i in range(batches):
        temp = data[i * batch_size: int(min((i + 1) * batch_size, data_size))]

        final_data[i] = [np.amax(temp), np.average(temp), np.amin(temp), 0]

        if not i == 0:
            final_data[i, 3] = final_data[i, 1] - final_data[i - 1, 1]

    x_axis = batch_size * np.arange(0, final_data.shape[0])

    plt.figure()
    plt.subplot(211)

    line_widths = [1, 2, 1, 0.5]
    line_colors = ['r', 'g', 'b', 'm']
    texts = ['max', 'data', 'min', 'der']
    for i in range(3):  # derivative out
        if batch_size == 1 and not i == 1:
            continue

        index = int((i + 5) * 0.1 * len(final_data[:, i]))
        plt.plot(x_axis, final_data[:, i], line_colors[i], linewidth=line_widths[i])
        plt.text(0.05 * len(final_data[:, i]), (i + 1) * 0.1 * np.amax(final_data[:, 0]),
                 texts[i], color=line_colors[i])

        # plt.annotate(texts[i],  xy=(x_axis[index], final_data[index, i]),
        #              xytext=(x_axis[index], final_data[index, i] + int(np.amax(final_data) * 0.4)),
        #              arrowprops=dict(facecolor=line_colors[i], shrink=0.05))

    # plt.plot(x_axis, final_data[:, 0], 'r', linewidth = 1)
    # plt.plot(x_axis, final_data[:, 1], 'g')
    # plt.plot(x_axis, final_data[:, 2], 'b', linewidth = 1)
    # plt.plot(x_axis, final_data[:, 3], 'm--', linewidth = 0.5)

    plt.grid(True)
    plt.title(ntpath.basename(file_name) + "(" + str(batch_size) + " batch size)")
    plt.ylabel("Reward")
    plt.xlabel("Episode")

    # reduced_data, ignored = ignore_low_values(data)
    # reduced_data, ignored = ignore_starting_rewards(data)
    reduced_data, ignored = data, 0
    STAT_GROUPS = 20
    MAX_VALUE = np.amax(reduced_data)
    # statistics
    stats = np.zeros((STAT_GROUPS))
    for i in reduced_data:
        index = int(i / ((MAX_VALUE + 1) / STAT_GROUPS))
        stats[index] += 1

    #stats *=100/len(data)
    x_axis = ((MAX_VALUE + 1) / STAT_GROUPS) * np.arange(STAT_GROUPS)
    plt.subplot(212)
    plt.plot(x_axis, stats, 'go-')
    # plt.axis([0, MAX_VALUE+1])
    plt.yscale("log")
    plt.grid(True)
    # plt.title("Statistics histogram")
    # plt.ylabel("%(ign "+ str(round(100*ignored/len(data)))+ '%)')
    plt.ylabel("Samples")
    plt.xlabel("Value")

    plt.show()

# unstested


def plot_surface(X, Y, Z):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1)

    t = plt.imshow(Z)

    t.set_cmap(cm.coolwarm)
    plt.colorbar()
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    # fig.colorbars(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


def ignore_starting_rewards(data, threshold=200):
    index = 0
    for i in range(len(data)):
        if data[i] >= threshold:
            index = i
            break
    return data[index:], index


def ignore_low_values(data, threshold=200):
    res = np.extract(data > 200, data)
    return res, len(data) - len(res)


def set_patameters_and_get_files():
    import argparse as arg

    parser = arg.ArgumentParser(description="Plot given reward files")
    parser.add_argument("file", type=str, nargs='*',
                        help="files to be plotted")
    parser.add_argument("-r", "--ratio", type=float,
                        help="batch to sample size ratio. Default: 0.01")
    parser.add_argument("-f", "--directory", type=str,
                        help="specify taret directory. Default: /")

    directory = parser.parse_args().directory
    if directory is not None:
        global DIRECTORY
        if directory[0] != '/':
            directory = '/' + directory
        DIRECTORY = directory
    # parser.add_argument("-e", "--extensions", type=str ,
    #                     help="Extension to b searched. Default: .txt")
    #
    # ext = parser.parse_args().extensions
    # if ext is not None:
    #     global EXTENSIONS
    #     EXTENSIONS = ext

    files = parser.parse_args().file
    if len(files) == 0:
        files = get_all_txt_files()

    rat = parser.parse_args().ratio
    if rat is not None:
        global BATCH_RATIO
        BATCH_RATIO = rat

    print(parser.parse_args().ratio)
    return files


def get_all_txt_files():
    from os import listdir
    from os.path import isfile, join, dirname, realpath, splitext

    mypath = dirname(realpath(__file__)) + DIRECTORY
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    txtfiles = []
    for f in onlyfiles:
        if splitext(f)[1] in EXTENSIONS:
            txtfiles.append(mypath + "/" + f)
    return txtfiles


if __name__ == '__main__':
    files = set_patameters_and_get_files()
    for f in files:
        try:
            plot_file(f)
        except ValueError as e:
            print(e)
