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
import svgwrite
import random

def load_data(data_dir='',model_dir=''):
    list_files = os.listdir(data_dir)
    list_files = sorted(list_files)
    chars, lbls = [], []
    chars_pts, LB = [], []
    data = []

    for file in list_files[:50]:

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

    chars_pts_before_clean = chars_pts_all
    lbls_before_clean = lbls_all

    chars_pts_after_clean, lbls_after_clean = remove_empty_labels(chars_pts_before_clean, lbls_before_clean)

    for w in range(len(chars_pts_after_clean)):
        for c in range(len(chars_pts_after_clean[w])):
            chars_pts_after_clean[w][c] = clean_double_points(chars_pts_after_clean[w][c])
            chars_pts_after_clean[w][c] = clean_one_point_strokes(chars_pts_after_clean[w][c])
            chars_pts_after_clean[w][c] = clean_redundant_points(chars_pts_after_clean[w][c], 0.9)



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

    # for i in range(len(chars_pts_normalized)):
    #     for c in range(len(chars_pts_normalized[i])):
    #         plot_char(chars_pts_normalized[i][c], lbls_all[i][c])

    strokes5 = []
    line_rebuild = []
    for i in range(len(Lines_normalized)):
        li = Lines_normalized[i]
        s5 = lines2strokes5(li)
        line_rebuild = strokes52lines(s5)

        # plot_char(lines2pts([line_rebuild])[0], 'bc')
        # diff = li - line_rebuild

        strokes5.append(s5)

    #draw_lines(strokes5)

    ALL_LINES = []
    ALL_LBLS = []
    for id_person in range(len(lbls_all)):
        ALL_LINES += strokes5[id_person]
        ALL_LBLS += lbls_all[id_person]

    #create_encode_decode_file(ALL_LBLS,model_dir)
    length = [np.max([len(x) for x in ALL_LINES]), np.mean([len(x) for x in ALL_LINES])]

    print('Max length: ' + str(length[0]) + '  Average length: ' + str(int(length[1])))

    # load encode file
    char2label = json.load(open(model_dir + 'encode_kanji.json'))
    label2char = {}
    for k, v in char2label.items():
        label2char[v] = k

    ALL_LBLS_ENCODED = file_to_word_ids(ALL_LBLS, char2label)

    stroke_train, stroke_val, label_train, label_val = train_test_split(ALL_LINES, ALL_LBLS_ENCODED, test_size=0.08,random_state=1)

    return stroke_train, stroke_val, label_train, label_val, label2char, char2label, length, ALL_LINES, ALL_LBLS_ENCODED

def file_to_word_ids(label, word_to_id):
    label_out = []
    for word in label:
        if word in word_to_id:
            label_out.append(word_to_id[word])
        else:
            print(word + 'not exist')
    return label_out

def plot_char(folder, char, lbl):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(len(char)):
        x,y = zip(*char[i])
        ax.plot(x,np.dot(y,-1))
    plt.axes().set_aspect('equal', 'datalim')

    if lbl[0] == '/':
        lbl = 'sur'
    name = '_'+lbl[0]+ '_' + str(random.randint(0,1000000)) + '.png'
    fig.savefig(folder + name)
    # plt.show()

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

def lines2strokes5(Lines_in):
    # Lines in: chars # [x1, x2 ,y1, y2]
    Chars = []
    for c in range(len(Lines_in)):         # char c
        Char = []
        for s in range(len(Lines_in[c])):  # stroke s
            for l in range(len(Lines_in[c][s])):

                dx = Lines_in[c][s][l][1] - Lines_in[c][s][l][0]
                dy = Lines_in[c][s][l][3] - Lines_in[c][s][l][2]

                if l == len(Lines_in[c][s]) - 1 and s == len(Lines_in[c]) - 1:   # end of char:
                    si = [0, 0, 1]
                else:
                    si = [1, 0, 0]

                Char.append([dx, dy] + si)

                if l == len(Lines_in[c][s]) - 1 and s != len(Lines_in[c]) - 1:   # end of stroke
                    dx = Lines_in[c][s + 1][0][0] - Lines_in[c][s][l][1]
                    dy = Lines_in[c][s + 1][0][2] - Lines_in[c][s][l][3]
                    si = [0, 1, 0]
                    Char.append([dx, dy] + si)

        Chars.append(Char)
    return Chars

def strokes52lines(s5):
    # stroke 5: chars # [dx,dy ,s1, s2, s3]
    Lines = []
    for c in range(len(s5)):         # char c
        Char = []
        stroke = []
        for s in range(1,len(s5[c])):  # stroke s
            if s == 0:
                x1 = 0
                x2 = s5[c][s][0]
                y1 = 0
                y2 = s5[c][s][1]
            else:
                x1 = sum([s5[c][l][0]  for l in range(s)])
                x2 = sum([s5[c][l][0]  for l in range(s+1)])
                y1 = sum([s5[c][l][1]  for l in range(s)])
                y2 = sum([s5[c][l][1]  for l in range(s+1)])
            if s5[c][s][2] == 1:
                stroke.append([x1,x2,y1,y2])
            else:
                Char.append(stroke)
                stroke = []


        Lines.append(Char)
    return Lines

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

        if deltax == 0:
            print('Found deltax equal 0')
            deltax = 1


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
    # Line ins: chars # [x1, x2 ,y1, y2]
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


def clean_double_points(char_pts):
    char_out = list(char_pts)


    for s in range(len(char_out)):
        index = []
        for p in range(len(char_out[s])-1):
            if char_out[s][p] == char_out[s][p + 1]:
                print('Found double points')
                index.append(p)
        if index:
            for inx in sorted(index, reverse=True):
                del char_out[s][inx]

    return char_out

def clean_one_point_strokes(char_pts):
    char_out = list(char_pts)

    index = []
    for s in range(len(char_out)):
        if len(char_out[s]) == 1:
            print('Found strokes with one point')
            index.append(s)

    if index:
        for inx in sorted(index, reverse=True):
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
    Tdis = 0.05*np.max([width,height])

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

def load_checkpoint(sess, checkpoint_path):
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)

def save_model(sess, model_save_path, global_step):
  saver = tf.train.Saver(tf.global_variables())
  checkpoint_path = os.path.join(model_save_path, 'vector')
  tf.logging.info('saving model %s.', checkpoint_path)
  tf.logging.info('global_step %i.', global_step)
  saver.save(sess, checkpoint_path, global_step=global_step)

def reset_graph():
  """Closes the current default session and resets the graph."""
  sess = tf.get_default_session()
  if sess:
    sess.close()
  tf.reset_default_graph()

def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  # result[-1, 2] = 1 # end char
  return result

def draw_lines(Lines):
    max_len = 200
    result = np.zeros(( max_len +1, 5), dtype=float)
    l = len(Lines[0][0])
    assert l <= max_len
    for i in range(l):
        result[i, 0:2] = Lines[0][0][i][0:2]
        result[i, 3] = Lines[0][0][i][3]
        result[i, 2] = Lines[0][0][i][2]  # 1 - result[i, 0:l, 2]
    result[l - 1, 4] = 1
    # put in the first token, as described in sketch-rnn methodology
    result[1:, :] = result[:-1, :]
    result[0, :] = 0
    result[0, 2] = 1
    result[0, 3] = 0
    result[0, 4] = 0

    q =to_normal_strokes(result)
    draw_strokes(q)



def get_bounds(data, factor=10):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)

def draw_strokes(data,
                 factor=0.2,
                 svg_fpath = 'sample/sample.svg',
                 blur_dev = 0,
                 stroke_width=1,
                 angle = 0,
                 sX = 0,
                 sY = 0):
  tf.gfile.MakeDirs(os.path.dirname(svg_fpath))
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_fpath, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))

  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"

  strokes = []
  stroke = []

  data = np.delete(data, 0, 0)

  for i in range(len(data)):
    stroke.append(data[i])
    if data[i,2] == 1:
      strokes.append(stroke)
      stroke = []

  deltax = 0
  deltay = 0
  for stroke in strokes:
      lift_pen = 1

      for point in stroke:
          if (lift_pen == 1):
              command = "m"
              lift_pen = 0
          elif (command != "l"):
              command = "l"
          else:
              command = ""
          x = float(point[0]) / factor
          y = float(point[1]) / factor
          p += command + str(x) + "," + str(y) + " "

          deltax += x
          deltay += y

      the_color = "black"
      # stroke_width = random.uniform(0.5, 1.4)
      dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
      p = "M%s,%s " % (abs_x + deltax, abs_y + deltay)

  # create simple filter to blur rectangle
  blur6_filter = dwg.defs.add(dwg.filter())
  blur6_filter.feGaussianBlur(in_='SourceGraphic', stdDeviation=blur_dev)
  g_f = dwg.add(dwg.g(filter=blur6_filter.get_funciri()))

  g_f.skewX(sX)
  g_f.skewY(sY)
  g_f.rotate(angle)
  dwg.save()
  #display(SVG(dwg.tostring()))