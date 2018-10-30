import skeleton_extraction as se
import utils as ut
from functools import partial
import matplotlib.pyplot as plt
from pycpd import deformable_registration
import numpy as np
import random
import time
import os

def visualize(iteration, error, X, Y, ax, draw):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red', label='Target: offline', marker="X")
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue', label='Source: online')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    if draw:
        plt.draw()
        plt.pause(0.001)


def offline_preprocess(folder, file):
    ske = se.ske_ext(folder + file, draw=False)

    ske = ut.clean_double_points(ske)

    ske = ut.clean_one_point_strokes(ske)

    ske = ut.clean_redundant_points([ske], 0.95)

    lines_before_normalize = ut.pts2lines([ske])

    # normalize
    lines_after_normalize = ut.normalize(lines_before_normalize)

    # convert back to verify
    ske_normed = ut.lines2pts(lines_after_normalize)[0][0]

    return ske_normed

def CPD(X,Y, draw = True):


    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0], draw=draw)

    reg = deformable_registration(**{'X': X, 'Y': Y})
    reg.register(callback)

    if draw:
        plt.show()

    return reg.TY


def similarity(X, Y):
    dist = [[np.linalg.norm(X[i] - Y[j]) for j in range(len(Y))] for i in range(len(X))]

    total_cost = 0
    for i in range(len(X)):
        index = np.argmin(dist[i])
        total_cost += dist[i][index]

    return total_cost

def prepare_online_ref():
    chars = 'abcdefghijklmnopqrstuvxywzABCDEFGHIJKLMNOPQRSTUVXYWZ'
    datadir = "/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Online/Data/preprocessed/"

    labels = []
    Y = []
    for char in chars:
        # import online reference

        ol_ref = ut.load_data(datadir, input_char=char)[0][0][0]

        # scale the reference first
        Y.append(np.asarray(ol_ref) * 1.46 + 1)

    return Y, chars

def prepare_offline_ske():

    # import offline skeleton
    folder = "/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Offline/data_png/"
    # file = "d_764-f.gnt_63.png"

    filelist = []
    # Set the directory you want to start from
    rootDir = folder
    for dirName, subdirList, fileList in os.walk(rootDir):
        # print('Found directory: %s' % dirName)
        for fname in fileList:
            # print('\t%s' % fname)
            filelist.append(dirName + "/" + fname)

    ske_normed = []
    labels = []
    for i in range(len(fileList)):
        ske_normed.append(offline_preprocess(folder, fileList[i]))
        labels.append(fileList[i][0])

    return ske_normed,labels

if __name__ == '__main__':

    dist = []
    GoodDetect = 0
    FalseDetect = 0

    ske_normeds, offline_labels = prepare_offline_ske()
    online_refs, online_labels = prepare_online_ref()


    for i in range(len(ske_normeds)):
        for j in range(len(online_refs)):

            Y = online_refs[j]
            X = np.asarray(ske_normeds[i])

            # reverse the coordinates for plot
            Y[:, 1] = Y[:, 1] * -1
            X[:, 1] = X[:, 1] * -1


            # Map online to offline
            new_Y = CPD(X, Y, draw=False)


            dist.append(similarity(X, new_Y))

        index = np.argmin(dist)

        if online_labels[index] == offline_labels[i]:
            print('Good extraction')
            GoodDetect += 1
        else:
            print('False extraction')
            FalseDetect += 1

        print('Predicted ' + str(online_labels[index]) + ' with cost = ' + str(np.min(dist)))
        print('Real char: ' + str(offline_labels[i]))

        dist = []