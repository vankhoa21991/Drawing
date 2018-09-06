from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf


def orthogonal(shape):
  """Orthogonal initilaizer."""
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)


def orthogonal_initializer(scale=1.0):
  """Orthogonal initializer."""
  def _initializer(shape, dtype=tf.float32,
                   partition_info=None):  # pylint: disable=unused-argument
    return tf.constant(orthogonal(shape) * scale, dtype)

  return _initializer


def lstm_ortho_initializer(scale=1.0):
  """LSTM orthogonal initializer."""
  def _initializer(shape, dtype=tf.float32,
                   partition_info=None):  # pylint: disable=unused-argument
    size_x = shape[0]
    size_h = shape[1] // 4  # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h]) * scale
    t[:, size_h:size_h * 2] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 2:size_h * 3] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 3:] = orthogonal([size_x, size_h]) * scale
    return tf.constant(t, dtype)

  return _initializer


class GRU(tf.contrib.rnn.RNNCell):
  """Implementation of a Gated Recurrent Unit (GRU) as described in [1].

  [1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.

  Arguments
  ---------
  input_dimensions: int
      The size of the input vectors (x_t).
  hidden_size: int
      The size of the hidden layer vectors (h_t).
  dtype: obj
      The datatype used for the variables and constants (optional).
  """

  def __init__(self, num_units,
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.9):
    self.hidden_size = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob

  @property
  def state_size(self):
    return self.hidden_size

  @property
  def output_size(self):
    return self.hidden_size

  def __call__(self, x_t, h_tm1, scope=None):
    with tf.variable_scope(scope or type(self).__name__):

      self.input_dimensions = x_t.get_shape().as_list()[1]

      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)

      # Weights for input vectors of shape (input_dimensions, hidden_size)
      self.Wr = tf.get_variable('Wr', [self.input_dimensions, self.hidden_size], initializer=w_init)
      self.Wz = tf.get_variable('Wz', [self.input_dimensions, self.hidden_size], initializer=w_init)
      self.Wh = tf.get_variable('Wh', [self.input_dimensions, self.hidden_size], initializer=w_init)

      # Weights for hidden vectors of shape (hidden_size, hidden_size)
      self.Ur = tf.get_variable('Ur', [self.hidden_size, self.hidden_size], initializer=h_init)
      self.Uz = tf.get_variable('Uz', [self.hidden_size, self.hidden_size], initializer=h_init)
      self.Uh = tf.get_variable('Uh', [self.hidden_size, self.hidden_size], initializer=h_init)

      # Biases for hidden vectors of shape (hidden_size,)
      self.br = tf.get_variable('br', [self.hidden_size],initializer=tf.constant_initializer(0.0))
      self.bz = tf.get_variable('bz', [self.hidden_size], initializer=tf.constant_initializer(0.0))
      self.bh = tf.get_variable('bh', [self.hidden_size], initializer=tf.constant_initializer(0.0))

      # Definitions of z_t and r_t
      z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
      r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

      # Definition of h~_t
      h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

      # Compute the next hidden state
      h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
      # Perform the scan operator
      # h_t = self.forward_pass(x_t, h_tm1)

      return h_t, h_t

class GRU_embedding(tf.contrib.rnn.RNNCell):
  """Implementation of a Gated Recurrent Unit (GRU) as described in [1].

  [1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.

  Arguments
  ---------
  input_dimensions: int
      The size of the input vectors (x_t).
  hidden_size: int
      The size of the hidden layer vectors (h_t).
  dtype: obj
      The datatype used for the variables and constants (optional).
  """

  def __init__(self, num_units,
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.9):
    self.hidden_size = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob

  @property
  def state_size(self):
    return self.hidden_size

  @property
  def output_size(self):
    return self.hidden_size

  def __call__(self, x_t, h_tm1, scope=None):
    with tf.variable_scope(scope or type(self).__name__):

      self.input_dimensions = x_t.get_shape().as_list()[1]

      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)

      # Weights for input vectors of shape (input_dimensions, hidden_size)
      self.Wr = tf.get_variable('Wr', [self.input_dimensions, self.hidden_size], initializer=w_init)
      self.Wz = tf.get_variable('Wz', [self.input_dimensions, self.hidden_size], initializer=w_init)
      self.Wh = tf.get_variable('Wh', [self.input_dimensions, self.hidden_size], initializer=w_init)

      # Weights for hidden vectors of shape (hidden_size, hidden_size)
      self.Ur = tf.get_variable('Ur', [self.hidden_size, self.hidden_size], initializer=h_init)
      self.Uz = tf.get_variable('Uz', [self.hidden_size, self.hidden_size], initializer=h_init)
      self.Uh = tf.get_variable('Uh', [self.hidden_size, self.hidden_size], initializer=h_init)

      # Biases for hidden vectors of shape (hidden_size,)
      self.br = tf.get_variable('br', [self.hidden_size],initializer=tf.constant_initializer(0.0))
      self.bz = tf.get_variable('bz', [self.hidden_size], initializer=tf.constant_initializer(0.0))
      self.bh = tf.get_variable('bh', [self.hidden_size], initializer=tf.constant_initializer(0.0))

      # Definitions of z_t and r_t
      z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
      r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

      # Definition of h~_t
      h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

      # Compute the next hidden state
      h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
      # Perform the scan operator
      # h_t = self.forward_pass(x_t, h_tm1)

      return h_t, h_t

class LSTMCell(tf.contrib.rnn.RNNCell):
  """Vanilla LSTM cell.
  Uses ortho initializer, and also recurrent dropout without memory loss
  (https://arxiv.org/abs/1603.05118)
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               use_recurrent_dropout=False,
               dropout_keep_prob=0.9):
    self.num_units = num_units
    self.forget_bias = forget_bias
    self.use_recurrent_dropout = use_recurrent_dropout
    self.dropout_keep_prob = dropout_keep_prob

  @property
  def state_size(self):
    return 2 * self.num_units

  @property
  def output_size(self):
    return self.num_units

  def get_output(self, state):
    unused_c, h = tf.split(state, 2, 1)
    return h

  def __call__(self, x, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      c, h = tf.split(state, 2, 1)

      x_size = x.get_shape().as_list()[1]

      w_init = None  # uniform

      h_init = lstm_ortho_initializer(1.0)

      # Keep W_xh and W_hh separate here as well to use different init methods.
      w_xh = tf.get_variable(
          'W_xh', [x_size, 4 * self.num_units], initializer=w_init)
      w_hh = tf.get_variable(
          'W_hh', [self.num_units, 4 * self.num_units], initializer=h_init)
      bias = tf.get_variable(
          'bias', [4 * self.num_units],
          initializer=tf.constant_initializer(0.0))

      concat = tf.concat([x, h], 1)
      w_full = tf.concat([w_xh, w_hh], 0)
      hidden = tf.matmul(concat, w_full) + bias

      i, j, f, o = tf.split(hidden, 4, 1)

      if self.use_recurrent_dropout:
        g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
      else:
        g = tf.tanh(j)

      new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

      return new_h, tf.concat([new_c, new_h], 1)  # fuk tuples.