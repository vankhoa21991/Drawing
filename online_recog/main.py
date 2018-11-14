import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import random
import time
import logging

from pycpd import deformable_registration,affine_registration,rigid_registration
from functools import partial
from gg_onlineHW_api import ggapi
from collections import Counter
from preprocess import *
from utils import *

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


def multi_char_process(offline_folder, online_folder):
    dist = []
    GoodDetect = 0
    FalseDetect = 0

    # ske_normeds, offline_labels = prepare_offline_ske(offline_folder)
    ske_normeds, offline_labels = prepare_mnist()
    online_refs, online_labels = prepare_online_ref(online_folder)

    for i in range(len(ske_normeds)):
        for j in range(len(online_refs)):
            Y = online_refs[j]
            X = np.asarray(ske_normeds[i])

            X_norm, Y_norm = normalize(X, Y)

            # reverse the coordinates for plot
            Y_norm[:, 1] = Y_norm[:, 1] * -1
            X_norm[:, 1] = X_norm[:, 1] * -1

            # Map online to offline
            new_Y = CPD(X_norm, Y_norm, draw=False)

            dist.append(similarity(X_norm, new_Y))

        index = np.argmin(dist)

        if online_labels[index].lower() == offline_labels[i].lower():
            print('Good extraction')
            GoodDetect += 1
        else:
            print('False extraction')
            FalseDetect += 1

        print('Predicted ' + str(online_labels[index]) + ' with cost = ' + str(np.min(dist)))
        print('Real char: ' + str(offline_labels[i]))
        print('\n')
        logging.info('Predicted ' + str(online_labels[index]) + ' with cost = ' + str(np.min(dist)))
        logging.info('Real char: ' + str(offline_labels[i]))
        logging.info('\n')
        dist = []
    print('Accuracy: ' + str(GoodDetect / (GoodDetect + FalseDetect)))
    return

def single_char_process(offline_folder, online_folder):
    # ske_normeds, offline_labels = prepare_offline_ske(offline_folder)
    ske_normeds, offline_labels = prepare_mnist()
    online_refs, online_labels = prepare_online_ref(online_folder)

    input_char = '9'
    ref_char = '4'

    i = random.choice([k for k, e in enumerate(offline_labels) if e == input_char])


    pred = []

    for ref_char in online_labels:
        j = [k for k, e in enumerate(online_labels) if e == ref_char]
        Y = online_refs[random.choice(j)]
        X = np.asarray(ske_normeds[i])

        fig = plt.figure(1)
        fig.add_axes([0, 0, 1, 1])
        ax = fig.axes[0]
        ax.scatter(X[:, 0], X[:, 1]* -1, color='red', label='Target: offline', marker="X")
        ax.scatter(Y[:, 0], Y[:, 1]* -1, color='blue', label='Source: online')
        fig.savefig(ref_char + 'to' + input_char + 'before_norm')
        ax.cla()

        X_norm, Y_norm = normalize(X, Y)

        fig = plt.figure(2)
        fig.add_axes([0, 0, 1, 1])
        ax = fig.axes[0]
        ax.scatter(X_norm[:, 0], X_norm[:, 1]* -1, color='red', label='Target: offline', marker="X")
        ax.scatter(Y_norm[:, 0], Y_norm[:, 1]* -1, color='blue', label='Source: online')
        fig.savefig(ref_char + 'to' + input_char +'after_norm')
        ax.cla()

        # reverse the coordinates for plot
        Y_norm[:, 1] = Y_norm[:, 1] * -1
        X_norm[:, 1] = X_norm[:, 1] * -1

        # Map online to offline
        new_Y = CPD(X_norm, Y_norm, draw=False)

        dist = similarity(X_norm, new_Y)
        print(dist)

        fig = plt.figure(3)
        fig.add_axes([0, 0, 1, 1])
        ax = fig.axes[0]
        ax.scatter(X_norm[:, 0], X_norm[:, 1], color='red', label='Target: offline', marker="X")
        ax.scatter(new_Y[:, 0], new_Y[:, 1], color='blue', label='Source: online')
        fig.savefig(ref_char + 'to' + input_char + 'after_fit' )
        ax.cla()

        pred_lb = ggapi(new_Y)
        pred.append(pred_lb)

    cnts = Counter(pred)
    # Get the maximum count
    maximum_cnt = max(cnts.values())
    # print all values that have the "maximum" count
    pred_final = [val for val, cnt in cnts.items() if cnt == maximum_cnt]

    if input_char in pred_final:
        print('Good extraction')

    else:
        print('False extraction')
    return


if __name__ == '__main__':
    
    server = False
    
    if server:
        offline_folder = "/mnt/DATA/lupin/Flaxscanner/Dataset/CASIA/CASIA_extracted/CASIA_offline/data_png/"
        online_folder = "/mnt/DATA/lupin/Flaxscanner/Dataset/CASIA/Drawing/"
    else:
        offline_folder = "/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Offline/data_png/"
        online_folder = "/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Online/Data/preprocessed/"
    logging.basicConfig(filename="logging.log", level=logging.INFO)

    single_char_process(offline_folder, online_folder)

    # multi_char_process(offline_folder, online_folder)

