import os
import preprocess as pres
import tensorflow as tf
import skeleton_extraction as se
from skimage.morphology import skeletonize
from skimage.util import invert
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


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
    if cover_percentage < 0.8:
        total_cost = 9999
    else:
        total_cost = sum(min_cost)  # - 10*cover_percentage

    return total_cost


def prepare_online_ref(datadir):
    chars = 'abcdefghijklmnopqrstuvxywzABCDEFGHIJKLMNOPQRSTUVXYWZ0123456789'
    chars = '0123456789'

    labels = []
    Y = []
    for char in chars:
        # import online reference

        ol_ref = pres.online_preprocess(datadir, input_char=char)[0][0]

        long_ref = interpolate(ol_ref)
        a = []
        for s in range(len(long_ref)):
            for i in range(len(long_ref[s])):
                a.append(long_ref[s][i])

        # scale the reference first
        Y.append(np.asarray(a))

    return Y, chars

def test_interpolate():


    timestamp = (0, 5, 10, 15, 30, 35, 40, 50, 55, 60)
    x_coords = (0, 10, 12, 13, 19, 13, 12, 19, 21, 25)
    y_coords = (0, 5, 10, 7, 2, 8, 15, 19, 14, 15)

    start_timestamp = min(timestamp)
    end_timestamp = max(timestamp)
    duration_seconds = (end_timestamp - start_timestamp)

    new_intervals = np.linspace(start_timestamp, end_timestamp, duration_seconds)

    new_x_coords = np.interp(new_intervals, timestamp, x_coords)
    new_y_coords = np.interp(new_intervals, timestamp, y_coords)

    plt.plot(x_coords, y_coords, 'o')
    plt.plot(new_x_coords, new_y_coords, '-x')
    plt.show()

def interpolate(ref, npts=3, draw = False):
    char = []
    for s in range(len(ref)):
        l = len(ref[s])
        timestamp = range(0,l*npts,npts)
        x_coords, y_coords = zip(*ref[s])

        start_timestamp = min(timestamp)
        end_timestamp = max(timestamp)
        duration_seconds = (end_timestamp - start_timestamp)

        new_intervals = np.linspace(start_timestamp, end_timestamp, duration_seconds)

        new_x_coords = np.interp(new_intervals, timestamp, x_coords)
        new_y_coords = np.interp(new_intervals, timestamp, y_coords)

        if draw:
            plt.plot(x_coords, y_coords, 'o')
            plt.plot(new_x_coords, new_y_coords, '-x')
            plt.show()

        stroke = []
        for p in range(len(new_x_coords)):
            stroke.append([new_x_coords[p],new_y_coords[p]])
        char.append(stroke)
    return char




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

    # for j in range(len(skes)):
    #     pres.plot_char('skeleton', [skes[j]], str(y_test[j]), draw=True)

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
