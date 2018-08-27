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
from keras.callbacks import Callback


class DataLoader(object):
  """Class for loading data."""

  def __init__(self, strokes='', charlabel='',args=''):

      self.strokes = strokes
      self.label=charlabel
      self.batch_size = args.batch_size  # minibatch size
      self.max_seq_length = args.max_seq_length  # N_max in sketch-rnn paper
      self.char2label = json.load(open(args.model_dir + 'encode_kanji.json'))
      self.vocabulary = len(self.char2label)
      self.skip_step = 2
      self.current_idx = 0

      self.pad_strokes = sequence.pad_sequences(strokes, maxlen=args.max_seq_length)
      # self.pad_strokes = np.transpose(self.pad_strokes,(0,2,1))

class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1
