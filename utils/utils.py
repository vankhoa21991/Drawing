import os
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_dir=''):
    list_files = os.listdir(data_dir)
    list_files = sorted(list_files)
    chars, lbls = [], []
    chars_pts, LB = [], []
    data = []
    for file in list_files[:20]:

        if file[-9:] == '_lbls.txt':
            file_name = file[:3]
            try:
                file_stroke = open(data_dir + file_name + '_stroke.txt', "rb")
                file_lable = open(data_dir + file_name + '_lbls.txt', "rb")
            except:
                break

            strokes = pickle.load(file_stroke)
            chars.append(strokes)
            chars_pts += strokes

            lbl = pickle.load(file_lable)
            lbls.append(lbl)
            LB += lbl


    return chars, lbls, chars_pts, LB



def plot_char(char):
    fig = plt.figure()
    ax = plt.axes()

    for i in range(len(char)):
        x,y = zip(*char[i])
        plt.plot(x,np.dot(y,-1))
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


def pts2lines(CHAR):
    # input: CHAR[char[stroke[x,y]]]
    Lines = []
    for char in CHAR:
        Line = []
        for s in range(len(char)):
            Stroke = []
            for i in range(len(char[s]) - 1):
                x1 = char[s][i][0]
                x2 = char[s][i + 1][0]
                y1 = char[s][i][1]
                y2 = char[s][i + 1][1]

                Stroke.append([x1, x2, y1, y2])

            Line.append(Stroke)

        Lines.append(Line)

    return Lines  # [x1, x2 ,y1, y2]


def lines2pts(Lines):
    CHAR = []
    for c in range(len(Lines)):  # char: LInes[c]
        Strokes = []
        for s in range(len(Lines[c])):  # stroke: Lines[c][s]
            Line = []
            for l in range(len(Lines[c][s])):  # line: Lines[c][s][l]
                x1 = Lines[c][s][l][0]
                x2 = Lines[c][s][l][1]
                y1 = Lines[c][s][l][2]
                y2 = Lines[c][s][l][3]

                if l == len(Lines[c][s]) - 1:
                    Line.append((x1, y1))
                    Line.append((x2, y2))
                else:
                    Line.append((x1, y1))
            Strokes.append(Line)
        CHAR.append(Strokes)
    return CHAR


def normalize(Lines):
    for c in range(len(Lines)):  # char: LInes[c]
        Length, dxL = [], []
        px, py = [], []
        for s in range(len(Lines[c])):  # stroke: Lines[c][s]

            for l in range(len(Lines[c][s])):  # line: Lines[c][s][l]
                x1 = Lines[c][s][l][0]
                x2 = Lines[c][s][l][1]
                y1 = Lines[c][s][l][2]
                y2 = Lines[c][s][l][3]

                leng = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                Length.append(leng)

                px.append(0.5 * leng * (x1 + x2))
                py.append(0.5 * leng * (y1 + y2))

        mux = np.sum(px) / np.sum(Length)
        muy = np.sum(py) / np.sum(Length)

        for s in range(len(Lines[c])):  # stroke: Lines[c][s]
            for l in range(len(Lines[c][s])):  # line: Lines[c][s][l]
                x1 = Lines[c][s][l][0]
                x2 = Lines[c][s][l][1]
                y1 = Lines[c][s][l][2]
                y2 = Lines[c][s][l][3]

                leng = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                dxL.append((1 /3) * leng * ((x2 - mux) ** 2 + (x1 - mux) ** 2 + (x1 - mux) * (x2 - mux)))
        deltax = np.sqrt(np.sum(dxL) / np.sum(Length))


        for s in range(len(Lines[c])):  # stroke: Lines[c][s]
            # normalize
            for l in range(len(Lines[c][s])):  # line: Lines[c][s][l]
                x1 = Lines[c][s][l][0]
                x2 = Lines[c][s][l][1]
                y1 = Lines[c][s][l][2]
                y2 = Lines[c][s][l][3]

                Lines[c][s][l][0] = (x1 - mux) / deltax
                Lines[c][s][l][1] = (x2 - mux) / deltax
                Lines[c][s][l][2] = (y1 - muy) / deltax
                Lines[c][s][l][3] = (y2 - muy) / deltax
    return Lines


def extract_line(Lines_in):
    # Lines: chars # [x1, x2 ,y1, y2]
    Lines = []
    for c in range(len(Lines_in)):
        Line = []
        for s in range(len(Lines_in[c])):

            for l in range(len(Lines_in[c][s])):

                x = Lines_in[c][s][l][0]
                y = Lines_in[c][s][l][2]
                dx = Lines_in[c][s][l][1] - Lines_in[c][s][l][0]
                dy = Lines_in[c][s][l][3] - Lines_in[c][s][l][2]
                s1 = 1
                s2 = 0

                Line.append([x, y, dx, dy, s1, s2])

                if s != len(Lines_in[c]) - 1:  # not last stroke
                    x = Lines_in[c][s][l][1]
                    y = Lines_in[c][s][l][3]
                    dx = Lines_in[c][s+1][0][0] - Lines_in[c][s][l][1]
                    dy = Lines_in[c][s+1][0][1] - Lines_in[c][s][l][3]
                    s1 = 0
                    s2 = 1
                    Line.append([x, y, dx, dy, s1, s2])

        Lines.append(Line)
    return Lines

def remove_empty_labels(chars, lbls):
    chars_out = list(chars)
    lbls_out = lbls[:]
    for i in range(len(lbls_out)):
        index = [k for k, x in enumerate(lbls_out[i]) if x == '']
        for j in sorted(index, reverse=True):
            del chars_out[i][j]
            del lbls_out[i][j]
    return chars_out, lbls_out

def clean_redundant_points(chars_pts):
    chars_out = list(chars_pts)

    for c in range(len(chars_out)):
        for s in range(len(chars_out[c])):
            index = []
            for p in range(len(chars_out[c][s])):
                if len(chars_out[c][s][p]) == 1:
                    print('Found strokes with one point')
                    index.append(p)
            if index:
                for inx in index:
                    del chars_out[c][s][inx]

    return chars_out