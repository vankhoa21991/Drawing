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


class GRU:
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

  def __init__(self, x_t, hidden_size, dtype=tf.float32):
    self.input_dimensions = x_t.get_shape().as_list()[2]
    self.hidden_size = hidden_size

    w_init = None  # uniform

    h_init = lstm_ortho_initializer(1.0)

    # Weights for hidden vectors of shape (hidden_size, hidden_size)


    self.Wr = tf.get_variable('Wr', [self.input_dimensions, self.hidden_size], initializer=w_init)
    self.Wz = tf.get_variable('Wz', [self.input_dimensions, self.hidden_size], initializer=w_init)
    self.Wh = tf.get_variable('Wh', [self.input_dimensions, self.hidden_size], initializer=w_init)


    # Weights for hidden vectors of shape (hidden_size, hidden_size)
    self.Ur = tf.get_variable('Ur', [self.hidden_size, self.hidden_size], initializer=h_init)
    self.Uz = tf.get_variable('Uz', [self.hidden_size, self.hidden_size], initializer=h_init)
    self.Uh = tf.get_variable('Uh', [self.hidden_size, self.hidden_size], initializer=h_init)

    self.br = tf.get_variable('br', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bz = tf.get_variable('bz', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bh = tf.get_variable('bh', [self.hidden_size], initializer=tf.constant_initializer(0.0))

    # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
    self.h_0 = tf.matmul(x_t[0, :, :], tf.zeros(dtype=tf.float32, shape=(self.input_dimensions, hidden_size)),
                         name='h_0')

    # Perform the scan operator
    self.h_t = tf.scan(self.forward_pass, x_t, initializer=self.h_0, name='h_t_transposed')


  def forward_pass(self, h_tm1, x_t):
    """Perform a forward pass.

    Arguments
    ---------
    h_tm1: np.matrix
        The hidden state at the previous timestep (h_{t-1}).
    x_t: np.matrix
        The input vector.
    """
    # Definitions of z_t and r_t
    z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
    r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

    # Definition of h~_t
    h_proposal = tf.tanh(tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

    # Compute the next hidden state
    h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

    return h_t

class GRU_embedding():

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

  def __init__(self, x_t,num_units, pen_dim = 400, embeding_size = 4020, c='', state = []):
    self.c = c
    self.hidden_size = num_units                            # size RNN cell
    self.input_dimensions = x_t.get_shape().as_list()[2]    # size pen = 5
    # self.batch_size = 1
    # self.seq_max_len = 1
    self.pen_dim = pen_dim                                  # size pen in higher dimension
    self.embed_dim = embeding_size

    w_init = None  # uniform

    h_init = lstm_ortho_initializer(1.0)

    # Weights for input vectors of shape (input_dimensions, hidden_size)
    self.Wd = tf.get_variable('Wd', [3, self.pen_dim], initializer=w_init)
    self.Ws = tf.get_variable('Ws', [2, self.pen_dim], initializer=w_init)
    self.Wr = tf.get_variable('Wr', [self.hidden_size, self.hidden_size], initializer=w_init)
    self.Wz = tf.get_variable('Wz', [self.hidden_size, self.hidden_size], initializer=w_init)
    self.Wh = tf.get_variable('Wh', [self.hidden_size, self.hidden_size], initializer=w_init)
    self.W = tf.get_variable('W', [self.hidden_size, self.hidden_size], initializer=w_init)
    self.Wo = tf.get_variable('Wo', [self.hidden_size, self.hidden_size], initializer=w_init)

    # Weights for hidden vectors of shape (hidden_size, hidden_size)
    self.Ur = tf.get_variable('Ur', [self.pen_dim, self.hidden_size], initializer=h_init)
    self.Uz = tf.get_variable('Uz', [self.pen_dim,self.hidden_size], initializer=h_init)
    self.U = tf.get_variable('U', [self.pen_dim, self.hidden_size], initializer=h_init)
    self.Uo = tf.get_variable('Uo', [self.pen_dim, self.hidden_size], initializer=h_init)

    # Biases for hidden vectors of shape (hidden_size,)
    self.bd = tf.get_variable('bd', [self.pen_dim], initializer=tf.constant_initializer(0.0))
    self.bs = tf.get_variable('bs', [self.pen_dim], initializer=tf.constant_initializer(0.0))
    self.br = tf.get_variable('br', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bz = tf.get_variable('bz', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bh = tf.get_variable('bh', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bo = tf.get_variable('bo', [self.hidden_size], initializer=tf.constant_initializer(0.0))

    self.Vr = tf.get_variable('Vr', [self.pen_dim, self.hidden_size], initializer=h_init)
    self.Vz = tf.get_variable('Vz', [self.pen_dim,self.hidden_size], initializer=h_init)
    self.V = tf.get_variable('V', [self.pen_dim, self.hidden_size], initializer=h_init)
    self.Vo = tf.get_variable('Vo', [self.pen_dim, self.hidden_size], initializer=h_init)

    self.Mr = tf.get_variable('Mr', [self.hidden_size, self.hidden_size], initializer=h_init)
    self.Mz = tf.get_variable('Mz', [self.hidden_size, self.hidden_size], initializer=h_init)
    self.M = tf.get_variable('M', [self.hidden_size, self.hidden_size], initializer=h_init)
    self.Mo = tf.get_variable('Mo', [self.hidden_size, self.hidden_size], initializer=h_init)

    # Put the time-dimension upfront for the scan operator
    # x_t = tf.transpose(x_t, [0, 2, 1], name='x_t')

    # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
    # self.h_0 = tf.matmul(x_t[0, :, :], tf.zeros(dtype=tf.float32, shape=(self.input_dimensions, 2*self.hidden_size)),
    #                      name='h_0')

    # Perform the scan operator
    self.step = tf.constant(0)

    self.out = tf.scan(self.forward_pass, x_t, initializer=state, name='h_t_transposed')

    # Transpose the result back
    # self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

  def forward_pass(self, h_tm1, x_t):
    """Perform a forward pass.

    Arguments
    ---------
    h_tm1: np.matrix
        The hidden state at the previous timestep (h_{t-1}).
    x_t: np.matrix
        The input vector.
    """
    h_tm1 = tf.reshape(h_tm1, (2,-1,500))[0,:,:]

    a = tf.nn.embedding_lookup(self.c, self.step)
    self.c_in = tf.reshape(a, (-1,500))

    self.step = tf.add(self.step,1)

    dt = x_t[:, :3]
    st = x_t[:, 3:]

    d_tp = tf.tanh(tf.matmul(dt,self.Wd) + self.bd)
    s_tp = tf.tanh(tf.matmul(st,self.Ws) + self.bs)

    z_t = tf.sigmoid(tf.matmul(h_tm1, self.Wz) + tf.matmul(d_tp, self.Uz) + \
                     tf.matmul(s_tp,self.Vz) + tf.matmul(self.c_in,self.Mz) + self.bz)
    r_t = tf.sigmoid(tf.matmul(h_tm1, self.Wr) + tf.matmul(d_tp, self.Ur) + \
                     tf.matmul(s_tp,self.Vr) + tf.matmul(self.c_in,self.Mr) + self.br)
    h_bar = tf.tanh(tf.matmul(tf.multiply(r_t,h_tm1),self.W) + tf.matmul(d_tp,self.U) + \
                    tf.matmul(s_tp, self.V) + tf.matmul(self.c_in, self.M) + self.bh)

    # Compute the next hidden state
    h_t = tf.multiply(z_t, h_tm1) + tf.multiply(1 - z_t, h_bar)
    o_t = tf.tanh(tf.matmul(h_t,self.Wo) + tf.matmul(d_tp,self.Uo) + tf.matmul(s_tp,self.Vo) + \
                  tf.matmul(self.c_in,self.Mo) + self.bo)

    return tf.concat([h_t, o_t],1)


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