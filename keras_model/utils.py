import os
import _pickle as pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import six
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import io

def load_data(data_dir='',model_dir=''):
    list_files = os.listdir(data_dir)
    list_files = sorted(list_files)
    chars, lbls = [], []
    chars_pts, LB = [], []
    data = []
    for file in list_files:

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

    chars_pts_all, lbls_all = chars, lbls

    # get first 5 people
    chars_pts_before_clean = chars_pts_all
    lbls_before_clean = lbls_all

    chars_pts_after_clean, lbls_after_clean = remove_empty_labels(chars_pts_before_clean, lbls_before_clean)

    for w in range(len(chars_pts_after_clean)):
        for c in range(len(chars_pts_after_clean[w])):
            chars_pts_after_clean[w][c] = clean_one_point_strokes(chars_pts_after_clean[w][c])

    for w in range(len(chars_pts_after_clean)):
        for c in range(len(chars_pts_after_clean[w])):
            chars_pts_after_clean[w][c] = clean_redundant_points(chars_pts_after_clean[w][c],0.99)



    Lines_normalized = []
    chars_pts_normalized = []
    for i in range(len(chars_pts_after_clean)):
        # points to lines
        lines_before_normalize = pts2lines(chars_pts_after_clean[i])

        # normalize
        lines_after_normalize = normalize(lines_before_normalize)

        # convert back to verify
        chars_pts_normalized.append(lines2pts(lines_after_normalize))

        Lines_normalized.append(lines_after_normalize)

    Lines_input = []
    for i in range(len(Lines_normalized)):
        Lines_input.append(extract_line(Lines_normalized[i]))

    ALL_LINES = []
    ALL_LBLS = []
    for i in range(len(lbls_all)):
        ALL_LINES += Lines_input[i]
        ALL_LBLS += lbls_all[i]

    #create_encode_decode_file(ALL_LBLS,model_dir)
    max_len = np.max([len(x) for x in ALL_LINES])

    # load encode file
    char2label = json.load(open(model_dir + 'encode_kanji.json'))
    label2char = {}
    for k, v in char2label.items():
        label2char[v] = k

    ALL_LBLS_ENCODED = file_to_word_ids(ALL_LBLS, char2label)

    stroke_train, stroke_val, label_train, label_val = train_test_split(ALL_LINES, ALL_LBLS_ENCODED, test_size=0.08,random_state=1)

    return stroke_train, stroke_val, label_train, label_val, label2char, char2label, max_len

def file_to_word_ids(label, word_to_id):
    label_out = []
    for word in label:
        if word in word_to_id:
            label_out.append(word_to_id[word])
        else:
            print(word + 'not exist')
    return label_out

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

def clean_one_point_strokes(char_pts):
    char_out = list(char_pts)

    index = []
    for s in range(len(char_out)):
        if len(char_out[s]) == 1:
            print('Found strokes with one point')
            index.append(s)
    if index:
        for inx in index:
            del char_out[inx]

    return char_out

def create_encode_decode_file(lbls,data_path):
    filelist = {}
    unique_lbls = list(set(lbls))
    for i in range(len(unique_lbls)):
        filelist[unique_lbls[i]]= i


    with io.open(data_path + 'encode_kanji.json', 'w', encoding='utf8') as json_file:
        json.dump(filelist, json_file, ensure_ascii=False)

def get_width_height(char_pts):
    px, py = [],[]
    for s in range(len(char_pts)):

        for p in range(len(char_pts[s])):
            px.append(char_pts[s][p][0])
            py.append(char_pts[s][p][1])
    return np.max(px) - np.min(px), np.max(py) - np.min(py)


def clean_redundant_points(char_pts, Tcos):

    # get width and height of a character
    width, height = get_width_height(char_pts)
    Tdis = 0.01*np.max([width,height])

    strokes = []
    for s in range(len(char_pts)):

        index = []
        for p in range(1,len(char_pts[s])-1):

            xi   = char_pts[s][p][0]
            xi_1 = char_pts[s][p-1][0]
            xi1  = char_pts[s][p+1][0]

            yi   = char_pts[s][p][1]
            yi_1 = char_pts[s][p-1][1]
            yi1  = char_pts[s][p+1][1]

            dxi = xi1 - xi
            dxi_1 = xi - xi_1

            dyi = yi1 - yi
            dyi_1 = yi - yi_1

            Tc = (dxi_1*dxi + dyi_1*dyi)/(np.sqrt(dxi_1**2 + dyi_1**2)*np.sqrt(dxi**2 + dyi**2))

            Td = np.sqrt((xi - xi_1)**2 + (yi - yi_1)**2)
            if Td < Tdis:
                index.append(p)
            elif Tc > Tcos:
                index.append(p)

        if index:
            for inx in sorted(index, reverse=True):
                del char_pts[s][inx]


    return char_pts
