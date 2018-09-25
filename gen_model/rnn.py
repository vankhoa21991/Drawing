from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
# pylint: disable=unused-import
from tensorflow.python.ops.gen_functional_ops import remote_call
# pylint: enable=unused-import
from tensorflow.python.ops.gen_functional_ops import symbolic_gradient
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

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

  def __init__(self, x_t,num_units, pen_dim = 300, embeding_size = 500, c='', state = [], out_dim=300):
    self.c = c
    self.hidden_size = num_units                            # size RNN cell
    self.input_dimensions = x_t.get_shape().as_list()[2]    # size pen = 5
    # self.batch_size = 1
    # self.seq_max_len = 1
    self.pen_dim = pen_dim                                  # size pen in higher dimension
    self.embed_dim = embeding_size
    self.out_dim = out_dim

    init = tf.truncated_normal_initializer(0,0.01)


    # Weights for input vectors of shape (input_dimensions, hidden_size)
    self.Wd = tf.get_variable('Wd', [3, self.pen_dim], initializer=init)
    self.Ws = tf.get_variable('Ws', [2, self.pen_dim], initializer=init)
    self.Wr = tf.get_variable('Wr', [self.hidden_size, self.hidden_size], initializer=init)
    self.Wz = tf.get_variable('Wz', [self.hidden_size, self.hidden_size], initializer=init)
    self.W = tf.get_variable('W', [self.hidden_size, self.hidden_size], initializer=init)
    self.Wo = tf.get_variable('Wo', [self.hidden_size, self.out_dim], initializer=init)

    # Weights for hidden vectors of shape (hidden_size, hidden_size)
    self.Ur = tf.get_variable('Ur', [self.pen_dim, self.hidden_size], initializer=init)
    self.Uz = tf.get_variable('Uz', [self.pen_dim,self.hidden_size], initializer=init)
    self.U = tf.get_variable('U', [self.pen_dim, self.hidden_size], initializer=init)
    self.Uo = tf.get_variable('Uo', [self.pen_dim, self.out_dim], initializer=init)

    # Biases for hidden vectors of shape (hidden_size,)
    self.bd = tf.get_variable('bd', [self.pen_dim], initializer=tf.constant_initializer(0.0))
    self.bs = tf.get_variable('bs', [self.pen_dim], initializer=tf.constant_initializer(0.0))
    self.br = tf.get_variable('br', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bz = tf.get_variable('bz', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bh = tf.get_variable('bh', [self.hidden_size], initializer=tf.constant_initializer(0.0))
    self.bo = tf.get_variable('bo', [self.out_dim], initializer=tf.constant_initializer(0.0))

    self.Vr = tf.get_variable('Vr', [self.pen_dim, self.hidden_size], initializer=init)
    self.Vz = tf.get_variable('Vz', [self.pen_dim,self.hidden_size], initializer=init)
    self.V = tf.get_variable('V', [self.pen_dim, self.hidden_size], initializer=init)
    self.Vo = tf.get_variable('Vo', [self.pen_dim, self.out_dim], initializer=init)

    self.Mr = tf.get_variable('Mr', [self.embed_dim, self.hidden_size], initializer=init)
    self.Mz = tf.get_variable('Mz', [self.embed_dim, self.hidden_size], initializer=init)
    self.M = tf.get_variable('M', [self.embed_dim, self.hidden_size], initializer=init)
    self.Mo = tf.get_variable('Mo', [self.embed_dim, self.out_dim], initializer=init)

    # Put the time-dimension upfront for the scan operator
    # x_t = tf.transpose(x_t, [0, 2, 1], name='x_t')

    # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
    # self.h_0 = tf.matmul(x_t[0, :, :], tf.zeros(dtype=tf.float32, shape=(self.input_dimensions, 2*self.hidden_size)),
    #                      name='h_0')

    # Perform the scan operator

    # x_in = tf.concat([x_t, tf.reshape(c, (32, -1, 500)) ], 1)

    self.out = scan(self.forward_pass, [x_t, c], initializer=state, name='h_t_transposed')

    # Transpose the result back
    # self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

  def forward_pass(self, h_tm1, x_t, c_in):
    """Perform a forward pass.

    Arguments
    ---------
    h_tm1: np.matrix
        The hidden state at the previous timestep (h_{t-1}).
    x_t: np.matrix
        The input vector.
    """
    h_tm1 = tf.reshape(h_tm1, (2,-1,self.hidden_size))[0,:,:]

    self.c_in = tf.reshape(c_in,(-1,self.embed_dim))

    # x_t, self.c_in = tf.split(t_in, [5, 500], 1)

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


def scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
         swap_memory=False, infer_shape=True, reverse=False, name=None):
  cs = elems[1]
  elems = elems[0]
  if not callable(fn):
    raise TypeError("fn must be callable.")

  input_is_sequence = nest.is_sequence(elems)
  input_flatten = lambda x: nest.flatten(x) if input_is_sequence else [x]
  def input_pack(x):
    return nest.pack_sequence_as(elems, x) if input_is_sequence else x[0]

  if initializer is None:
    output_is_sequence = input_is_sequence
    output_flatten = input_flatten
    output_pack = input_pack
  else:
    output_is_sequence = nest.is_sequence(initializer)
    output_flatten = lambda x: nest.flatten(x) if output_is_sequence else [x]
    def output_pack(x):
      return (nest.pack_sequence_as(initializer, x)
              if output_is_sequence else x[0])

  elems_flat = input_flatten(elems)
  cs_flat = input_flatten(cs)

  in_graph_mode = not context.executing_eagerly()
  with ops.name_scope(name, "scan", elems_flat):
    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode:
      # Any get_variable calls in fn will cache the first call locally
      # and not issue repeated network I/O requests for each iteration.
      varscope = vs.get_variable_scope()
      varscope_caching_device_was_none = False
      if varscope.caching_device is None:
        # TODO(ebrevdo): Change to using colocate_with here and in other
        # methods.
        varscope.set_caching_device(lambda op: op.device)
        varscope_caching_device_was_none = True

    # Convert elems to tensor array.
    elems_flat = [
        ops.convert_to_tensor(elem, name="elem") for elem in elems_flat]

    cs_flat = [
      ops.convert_to_tensor(c_, name="c") for c_ in cs_flat]

    # Convert elems to tensor array. n may be known statically.
    n = elems_flat[0].shape[0].value or array_ops.shape(elems_flat[0])[0]

    # TensorArrays are always flat
    elems_ta = [
        tensor_array_ops.TensorArray(dtype=elem.dtype, size=n,
                                     dynamic_size=False,
                                     infer_shape=True)
        for elem in elems_flat]

    cs_ta = [
      tensor_array_ops.TensorArray(dtype=c_.dtype, size=n,
                                   dynamic_size=False,
                                   infer_shape=True)
      for c_ in cs_flat]

    # Unpack elements
    elems_ta = [
        elem_ta.unstack(elem) for elem_ta, elem in zip(elems_ta, elems_flat)]

    cs_ta = [
      c_ta.unstack(c) for c_ta, c in zip(cs_ta, cs_flat)]

    if initializer is None:
      a_flat = [elem.read(n - 1 if reverse else 0) for elem in elems_ta]
      i = constant_op.constant(1)
    else:
      initializer_flat = output_flatten(initializer)
      a_flat = [ops.convert_to_tensor(init) for init in initializer_flat]
      i = constant_op.constant(0)

    # Create a tensor array to store the intermediate values.
    accs_ta = [
        tensor_array_ops.TensorArray(
            dtype=init.dtype, size=n,
            element_shape=init.shape if infer_shape else None,
            dynamic_size=False,
            infer_shape=infer_shape)
        for init in a_flat]

    if initializer is None:
      accs_ta = [acc_ta.write(n - 1 if reverse else 0, a)
                 for (acc_ta, a) in zip(accs_ta, a_flat)]

    def compute(i, a_flat, tas):
      """The loop body of scan.
      Args:
        i: the loop counter.
        a_flat: the accumulator value(s), flattened.
        tas: the output accumulator TensorArray(s), flattened.
      Returns:
        [i + 1, a_flat, tas]: the updated counter + new accumulator values +
          updated TensorArrays
      Raises:
        TypeError: if initializer and fn() output structure do not match
        ValueType: if initializer and fn() output lengths do not match
      """
      packed_elems = input_pack([elem_ta.read(i) for elem_ta in elems_ta])
      packed_cs = input_pack([c_ta.read(i) for c_ta in cs_ta])
      packed_a = output_pack(a_flat)
      a_out = fn(packed_a, packed_elems, packed_cs)
      nest.assert_same_structure(
          elems if initializer is None else initializer, a_out)
      flat_a_out = output_flatten(a_out)
      tas = [ta.write(i, value) for (ta, value) in zip(tas, flat_a_out)]
      if reverse:
        next_i = i - 1
      else:
        next_i = i + 1
      return (next_i, flat_a_out, tas)

    if reverse:
      initial_i = n - 1 - i
      condition = lambda i, _1, _2: i >= 0
    else:
      initial_i = i
      condition = lambda i, _1, _2: i < n
    _, _, r_a = control_flow_ops.while_loop(
        condition, compute, (initial_i, a_flat, accs_ta),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop, swap_memory=swap_memory,
        maximum_iterations=n)

    results_flat = [r.stack() for r in r_a]

    n_static = elems_flat[0].get_shape().with_rank_at_least(1)[0]
    for elem in elems_flat[1:]:
      n_static.merge_with(elem.get_shape().with_rank_at_least(1)[0])
    for r in results_flat:
      r.set_shape(tensor_shape.TensorShape(n_static).concatenate(
          r.get_shape()[1:]))

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return output_pack(results_flat)

def super_linear(x,
                 output_size,
                 scope=None,
                 reuse=False,
                 init_w='ortho',
                 weight_start=0.0,
                 use_bias=True,
                 bias_start=0.0,
                 input_size=None):
  """Performs linear operation. Uses ortho init defined earlier."""
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or 'linear'):
    if reuse is True:
      tf.get_variable_scope().reuse_variables()

    w_init = None  # uniform
    if input_size is None:
      x_size = shape[1]
    else:
      x_size = input_size
    if init_w == 'zeros':
      w_init = tf.constant_initializer(0.0)
    elif init_w == 'constant':
      w_init = tf.constant_initializer(weight_start)
    elif init_w == 'gaussian':
      w_init = tf.random_normal_initializer(stddev=weight_start)
    elif init_w == 'ortho':
      w_init = lstm_ortho_initializer(1.0)

    w = tf.get_variable(
        'super_linear_w', [x_size, output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable(
          'super_linear_b', [output_size],
          tf.float32,
          initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)