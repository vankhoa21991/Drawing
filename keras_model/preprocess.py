from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import os
# import svgwrite
# from IPython.display import SVG, display
import six
from six.moves import cStringIO as StringIO
from six.moves import xrange
import itertools
from keras.utils import to_categorical
import json


    list_files = os.listdir(data_dir)
      list_files = sorted(list_files)
      chars, lbls = [], []
      chars_pts, LB = [], []
      data = []
      for file in list_files[:6]:

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

      chars_pts_after_clean = clean_redundant_points(chars_pts_after_clean)

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

      create_encode_decode_file(ALL_LBLS)
      max_len = np.max([len(x) for x in ALL_LINES])
