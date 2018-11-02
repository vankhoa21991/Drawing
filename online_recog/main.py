import numpy as np
import random
import time
import os
import logging

from preprocess import *
from utils import *

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

    i = offline_labels.index('2')
    j = online_labels.index('2')

    Y = online_refs[j]
    X = np.asarray(ske_normeds[i])

    X_norm, Y_norm = normalize(X, Y)

    # reverse the coordinates for plot
    Y_norm[:, 1] = Y_norm[:, 1] * -1
    X_norm[:, 1] = X_norm[:, 1] * -1

    # Map online to offline
    new_Y = CPD(X_norm, Y_norm, draw=True)

    dist = similarity(X_norm, new_Y)
    print(dist)
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


    # single_char_process(offline_folder, online_folder)

    multi_char_process(offline_folder, online_folder)

