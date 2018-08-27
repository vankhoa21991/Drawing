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

  def __init__(self,
               strokes='', charlabel='',
               batch_size=100,
               max_seq_length=317,
               num_steps = 1000):

      self.strokes = strokes
      self.label=charlabel
      self.batch_size = batch_size  # minibatch size
      self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
      self.num_steps = num_steps
      self.char2label = json.load(open('/mnt/DATA/lupin/Drawing/keras_model/data/encode_kanji.json'))
      self.vocabulary = len(self.char2label)
      self.skip_step = 2
      self.current_idx = 0

      self.pad_strokes = sequence.pad_sequences(strokes, maxlen=max_seq_length)
      # self.pad_strokes = np.transpose(self.pad_strokes,(0,2,1))

  def _get_batch_from_indices(self, indices):
      """Given a list of indices, return the potentially augmented batch."""
      x_batch = []
      seq_len = []
      for idx in range(len(indices)):
          i = indices[idx]
          data = self.strokes[i]
          data_copy = np.copy(data)

          x_batch.append(data_copy)
          length = len(data_copy)
          seq_len.append(length)
      seq_len = np.array(seq_len, dtype=int)
      # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
      return x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len

  def random_batch(self):
      """Return a randomised portion of the training data."""
      idx = np.random.permutation(range(0, len(self.strokes)))[0:self.batch_size]
      return self._get_batch_from_indices(idx)

  def get_batch(self, idx):
      """Get the idx'th batch from the dataset."""
      assert idx >= 0, "idx must be non negative"
      assert idx < self.num_batches, "idx must be less than the number of batches"
      start_idx = idx * self.batch_size
      indices = range(start_idx, start_idx + self.batch_size)
      return self._get_batch_from_indices(indices)

  def pad_batch(self, batch, max_len):
      """Pad the batch to be stroke-5 bigger format as described in paper."""
      result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
      assert len(batch) == self.batch_size
      for i in range(self.batch_size):
          l = len(batch[i])
          assert l <= max_len
          result[i, 0:l, 0:2] = batch[i][:, 0:2]
          result[i, 0:l, 3] = batch[i][:, 2]
          result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
          result[i, l:, 4] = 1
          # put in the first token, as described in sketch-rnn methodology
          result[i, 1:, :] = result[i, :-1, :]
          result[i, 0, :] = 0
          result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
          result[i, 0, 3] = self.start_stroke_token[3]
          result[i, 0, 4] = self.start_stroke_token[4]

class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1
