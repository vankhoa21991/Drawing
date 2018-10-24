import skeleton_extraction as se
import utils as ut
from functools import partial
import matplotlib.pyplot as plt
from pycpd import deformable_registration
import numpy as np
import random
import time

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red', label='Target', marker="X")
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)



if __name__ == '__main__':
    datadir = "/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Online/Data/preprocessed/"
    # import online reference
    ol_ref = ut.load_data(datadir)[0][0][0]

    # import offline skeleton
    folder = "/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Offline/data_png/"
    file = "b_735-f.gnt_63.png"

    ske = se.ske_ext(folder + file, draw = False)

    ske = ut.clean_double_points(ske)

    ske = ut.clean_one_point_strokes(ske)

    ske = ut.clean_redundant_points([ske], 0.95)

    lines_before_normalize = ut.pts2lines([ske])

    # normalize
    lines_after_normalize = ut.normalize(lines_before_normalize)

    # convert back to verify
    ske_normed = ut.lines2pts(lines_after_normalize)[0][0]

    Y = np.asarray(ol_ref)*1.5
    X = np.asarray(ske_normed)*1
    Y[:, 1] = Y[:, 1] * -1
    X[:, 1] = X[:, 1] * -1
    # Map online to offline
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = deformable_registration(**{'X': X, 'Y': Y})
    reg.register(callback)
    plt.show()
