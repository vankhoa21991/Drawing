from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
from keras.preprocessing import sequence
import tensorflow as tf
import os
import six
from six.moves import cStringIO as StringIO
from six.moves import xrange
import itertools


class DataLoader(object):
  """Class for loading model."""

  def __init__(self,
               strokes, charlabel='',
               batch_size=100,
               max_seq_length=250,
               limit=1000,
               embedding_len = 500,
               trained_embedding=False,
               vocabulary = 0):
    self.batch_size = batch_size  # minibatch size
    self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
    # Removes large gaps in the model. x and y offsets are clamped to have
    # absolute value no greater than this limit.
    self.limit = limit
    self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
    # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
    # sorted by size)
    self.strokes = strokes
    self.charlabel = charlabel
    self.num_batches = int(len(charlabel) / self.batch_size)
    self.pad_strokes = sequence.pad_sequences(strokes, maxlen=max_seq_length, dtype='float')


    if trained_embedding == True:
        self.embedding_matrix = np.random.uniform(low=0, high=1, size=(vocabulary, embedding_len))
    else:
        self.embedding_matrix = np.random.uniform(low=0, high=1, size=(vocabulary, embedding_len ))


  def _get_batch_from_indices(self, indices):
    """Given a list of indices, return the potentially augmented batch."""
    x_batch = []
    seq_len = []
    embed_vec = []
    for idx in range(len(indices)):
      i = indices[idx]
      data = self.strokes[i]
      data_copy = np.copy(data)
      x_batch.append(data_copy)
      embed_vec.append(self.charlabel[i])
      length = len(data_copy)
      seq_len.append(length)
    seq_len = np.array(seq_len, dtype=int)
    # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
    return x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len, embed_vec

  def random_batch(self):
    """Return a randomised portion of the training model."""
    idx = np.random.permutation(range(0, len(self.strokes)))[0:self.batch_size]
    # idx = list(range(0,32,1))
    return self._get_batch_from_indices(idx)

  def pad_batch(self, batch, max_len):
    """Pad the batch to be stroke-5 bigger format as described in paper."""
    result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
    assert len(batch) == self.batch_size
    for i in range(self.batch_size):
      l = len(batch[i])
      assert l <= max_len
      result[i, 0:l, 0:2] = batch[i][:, 0:2]
      result[i, 0:l, 3] = batch[i][:, 3]
      result[i, 0:l, 2] = batch[i][:, 2] #1 - result[i, 0:l, 2]
      result[i, l-1, 4] = 1
      # put in the first token, as described in sketch-rnn methodology
      result[i, 1:, :] = result[i, :-1, :]
      result[i, 0, :] = 0
      result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
      result[i, 0, 3] = self.start_stroke_token[3]
      result[i, 0, 4] = self.start_stroke_token[4]
    return result

  def get_batch(self, idx):
    """Get the idx'th batch from the dataset."""
    assert idx >= 0, "idx must be non negative"
    assert idx < self.num_batches, "idx must be less than the number of batches"
    start_idx = idx * self.batch_size
    indices = range(start_idx, start_idx + self.batch_size)
    return self._get_batch_from_indices(indices)

  def random_sample(self):
    """Return a random sample, in stroke-3 format as used by draw_strokes."""
    idx = np.random.permutation(range(0, len(self.strokes)))[0:1]
    return self._get_batch_from_indices(idx)