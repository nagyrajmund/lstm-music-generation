import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import csv
import numpy as np

sns.set()

def moving_average(y, N=1):
    y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode='valid')
    return y_smooth

def plot_diagram(x, y, label, title="", show=False, file_name=None, scale=1, N=1):
    x = [scale * elem for elem in x]
    plt.plot(x, moving_average(y, N), label=label)
    matplotlib.rcParams.update({'font.size': 72})
    plt.xlabel("steps", horizontalalignment='right', x=1.0)
    plt.title(title)
    plt.legend()
    plt.tight_layout(pad=0.1)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)

    if show:
        plt.show()
    if file_name is not None:
        fig.savefig(file_name)

def open_csv(file_name, header=True):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        if header:
            next(reader, None)
        
        x, y = [], []
        for row in reader:
            x.append(float(row[1]))
            y.append(float(row[2]))
    return x, y

def plot_multiple_diagrams(file_names, labels, title, out_img, scales=None, N=1):
    nr_of_files = len(file_names)
    assert nr_of_files == len(labels)
    if scales is not None:
        assert nr_of_files == len(scales)
    else:
        scales = [1] * nr_of_files

    for i, (file_name, label, scale) in enumerate(zip(file_names, labels, scales)):
        x, y = open_csv(file_name)
        if i == nr_of_files - 1:
            plot_diagram(x, y, label, title, show=True, file_name=out_img, scale=scale, N=N)
        else:
            plot_diagram(x, y, label, title, scale=scale, N=N)