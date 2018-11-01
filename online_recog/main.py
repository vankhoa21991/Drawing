import skeleton_extraction as se
import utils as ut
from functools import partial
import matplotlib.pyplot as plt
from pycpd import deformable_registration,affine_registration,rigid_registration
import numpy as np
import random
import time
import os
import logging

def visualize(iteration, error, X, Y, ax, draw):
    if draw:
        plt.cla()
        ax.scatter(X[:,0] ,  X[:,1], color='red', label='Target: offline', marker="X")
        ax.scatter(Y[:,0] ,  Y[:,1], color='blue', label='Source: online')
        plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
    
        plt.draw()
        plt.pause(0.001)

def normalize(X,Y):
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
    scale_ratio = min(lx/ly,wx/wy)

    # Y_norm = Y*scale_ratio

    for i in range(len(Y_norm)):
        Y_norm[i][0] = Y[i][0]* scale_ratio + x_move
        Y_norm[i][1] = Y[i][1]* scale_ratio + y_move

    return X_norm,Y_norm

def CPD(X,Y, draw = True):
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

def similarity(X, Y):
    # dist = [[np.linalg.norm(X[i] - Y[j]) for j in range(len(Y))] for i in range(len(X))]

    cost_matrix = np.zeros((len(X), len(Y)), dtype=float)

    for i in range(len(X)):
        for j in range(len(Y)):
            cost_matrix[i, j] = np.linalg.norm(X[i] - Y[j])

    min_cost = 0
    ix = []
    for i in range(len(X)):
        index = np.argmin(cost_matrix[i,:])
        min_cost += cost_matrix[i,index]
        ix.append(index)

    cover_percentage = len(set(ix))/len(Y)
    if cover_percentage < 0.97:
        total_cost = 9999
    else:
        total_cost = min_cost #- 10*cover_percentage

    return total_cost

def prepare_online_ref(datadir):
    chars = 'abcdefghijklmnopqrstuvxywzABCDEFGHIJKLMNOPQRSTUVXYWZ'

    labels = []
    Y = []
    for char in chars:
        # import online reference

        ol_ref = ut.load_data(datadir, input_char=char)[0][0]


        a = []
        for s in range(len(ol_ref)):
            for i in range(len(ol_ref[s])):
                a.append(ol_ref[s][i])

        # scale the reference first
        Y.append(np.asarray(a))

    return Y, chars

def prepare_offline_ske(folder):

    # import offline skeleton
    #file = "d_764-f.gnt_63.png"
    
    
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
    for i in range(len(filelist)-1700):
        ske_normed.append(offline_preprocess(folder, filelist[i]))
        labels.append(filelist[i][0])

    return ske_normed,labels

def process(offline_folder, online_folder):
    dist = []
    GoodDetect = 0
    FalseDetect = 0

    ske_normeds, offline_labels = prepare_offline_ske(offline_folder)

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

def check_mapping(offline_folder, online_folder):
    ske_normeds, offline_labels = prepare_offline_ske(offline_folder)
    online_refs, online_labels = prepare_online_ref(online_folder)

    i = offline_labels.index('B')
    j = online_labels.index('B')

    Y = online_refs[j]
    X = np.asarray(ske_normeds[i])

    X_norm, Y_norm = normalize(X, Y)

    # reverse the coordinates for plot
    Y_norm[:, 1] = Y_norm[:, 1] * -1
    X_norm[:, 1] = X_norm[:, 1] * -1

    # Map online to offline
    new_Y = CPD(X_norm, Y_norm, draw=False)

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


    # check_mapping(offline_folder, online_folder)

    process(offline_folder, online_folder)

