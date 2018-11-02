import preprocess as pres
from functools import partial
import matplotlib.pyplot as plt
from pycpd import deformable_registration,affine_registration,rigid_registration
import tensorflow as tf
import skeleton_extraction as se
from skimage.morphology import skeletonize
from skimage.util import invert
import numpy as np

def visualize(iteration, error, X, Y, ax, draw):
    if draw:
        plt.cla()
        ax.scatter(X[:, 0], X[:, 1], color='red', label='Target: offline', marker="X")
        ax.scatter(Y[:, 0], Y[:, 1], color='blue', label='Source: online')
        plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')

        plt.draw()
        plt.pause(0.001)

def CPD(X, Y, draw=True):
    if draw:
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(visualize, ax=fig.axes[0], draw=draw)
    else:
        callback = []

    reg = deformable_registration(**{'X': X, 'Y': Y})
    reg.register(callback)

    if draw:
        plt.show()

    return reg.TY

def normalize(X, Y):
    X_norm = X[:]
    Y_norm = Y[:]

    xX = [i[0] for i in X]
    yX = [i[1] for i in X]

    xY = [i[0] for i in Y]
    yY = [i[1] for i in Y]

    lx = max(yX) - min(yX)
    wx = max(xX) - min(xX)
    x_center_X = np.mean(xX)
    y_center_X = np.mean(yX)

    ly = max(yY) - min(yY)
    wy = max(xY) - min(xY)
    x_center_Y = np.mean(xY)
    y_center_Y = np.mean(yY)

    x_move = x_center_X - x_center_Y
    y_move = y_center_X - y_center_Y
    scale_ratio = min(lx / ly, wx / wy)

    # Y_norm = Y*scale_ratio

    for i in range(len(Y_norm)):
        Y_norm[i][0] = Y[i][0] * scale_ratio + x_move
        Y_norm[i][1] = Y[i][1] * scale_ratio + y_move

    return X_norm, Y_norm

def similarity(X, Y):
    # dist = [[np.linalg.norm(X[i] - Y[j]) for j in range(len(Y))] for i in range(len(X))]

    cost_matrix = np.zeros((len(X), len(Y)), dtype=float)

    for i in range(len(X)):
        for j in range(len(Y)):
            cost_matrix[i, j] = np.linalg.norm(X[i] - Y[j])

    min_cost = []
    ix = []
    for i in range(len(X)):
        index = np.argmin(cost_matrix[i, :])

        min_cost.append(cost_matrix[i, index])
        ix.append(index)

    max_dist = max(min_cost)
    cover_percentage = len(set(ix)) / len(Y)
    if cover_percentage < 0.97 and max_dist > 5:
        total_cost = 9999
    else:
        total_cost = sum(min_cost)  # - 10*cover_percentage

    return total_cost


def prepare_online_ref(datadir):
    chars = 'abcdefghijklmnopqrstuvxywzABCDEFGHIJKLMNOPQRSTUVXYWZ'
    chars = '0123456789'

    labels = []
    Y = []
    for char in chars:
        # import online reference

        ol_ref = pres.online_preprocess(datadir, input_char=char)[0][0]

        a = []
        for s in range(len(ol_ref)):
            for i in range(len(ol_ref[s])):
                a.append(ol_ref[s][i])

        # scale the reference first
        Y.append(np.asarray(a))

    return Y, chars


def prepare_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    skes = []
    labels = []
    for i in range(len(x_test[:100])):
        labels.append(str(y_test[i]))
        img_bi = x_test[i] < 200

        width = len(img_bi)
        height = len(img_bi[0])

        # Invert the horse image
        image = invert(img_bi)

        # perform skeletonization
        skeleton = skeletonize(image)

        ske = []
        for x in range(width):
            for y in range(height):
                if skeleton[x, y] == True:
                    ske.append([y, x])

        # ske = ut.clean_double_points(ske)
        #
        # ske = ut.clean_one_point_strokes(ske)
        #
        # ske = ut.clean_redundant_points([ske], 0.95)

        lines_before_normalize = pres.pts2lines([[ske]])

        # normalize
        lines_after_normalize = pres.normalize(lines_before_normalize)

        # convert back to verify
        ske_normed = pres.lines2pts(lines_after_normalize)[0][0]

        skes.append(ske_normed)

    return skes, labels


def prepare_offline_ske(folder):
    # import offline skeleton
    # file = "d_764-f.gnt_63.png"

    filelist = []
    # Set the directory you want to start from
    rootDir = folder
    for dirName, subdirList, fileList in os.walk(rootDir):
        # print('Found directory: %s' % dirName)
        for fname in fileList:
            # print('\t%s' % fname)
            filelist.append(fname)

    ske_normed = []
    labels = []
    for i in range(len(filelist) - 1700):
        ske_normed.append(pres.offline_preprocess(folder, filelist[i]))
        labels.append(filelist[i][0])

    return ske_normed, labels
