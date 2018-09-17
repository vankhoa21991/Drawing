from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# internal imports

import numpy as np
import tensorflow as tf

import rnn

class Generation_model(object):
  def __init__(self, args, gpu_mode=True, reuse=False, vocabulary = 4020):
      with tf.variable_scope('vector_rnn', reuse=reuse):
        self.vocab = vocabulary
        self.build_model(args)


  def build_model(self, args):
    """Define model architecture."""

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None,], name='seq_len')
    self.input_data = tf.placeholder(dtype=tf.float32,shape=[None, None, 5], name='input')

    self.index_chars = tf.placeholder(dtype=tf.int32, shape=[None,], name='char_index')

    # The target/expected vectors of strokes
    self.output_x = self.input_data[:, 1:args.max_seq_len + 1, :]
    # vectors of strokes to be fed to decoder (same as above, but lagged behind
    # one step to include initial dummy value of (0, 0, 1, 0, 0))
    self.input_x = self.input_data[:, :args.max_seq_len, :]

    # cell_fn = rnn.LSTMCell
    # cell_fn = rnn.GRU

    self.embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab, args.embedding_len], initializer=None)

    chars = tf.nn.embedding_lookup(self.embedding_matrix, self.index_chars)

    self.initial_state = tf.placeholder(shape=[None, args.out_dim + args.hidden_size], dtype=tf.float32, name='initial_state')

    # if args.dropout_rate > 0:
    #   cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args.dropout_rate)

    self.cell = rnn.GRU_embedding(x_t=self.input_x,
                                  num_units=args.hidden_size,
                                  c = chars,
                                  state = self.initial_state,
                                  pen_dim=args.pen_dim,
                                  out_dim = args.out_dim)
    # self.cell = rnn.GRU(x_t=self.input_x, hidden_size=args.hidden_size)
    # self.initial_state = self.cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)


    output = tf.reshape(self.cell.out, (2,-1, args.hidden_size))[1,:,:]
    output = tf.reshape(output, [-1, args.out_dim])
    # last_state = tf.reshape(self.cell.out, (2, -1, 500))[0, :, :]

    self.num_mixture = args.num_mixture

    # TODO(deck): Better understand this comment.
    # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
    # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
    # mixture weight/probability (Pi_k)
    n_direction = (args.num_mixture * 5)
    n_state = 3

    with tf.variable_scope('RNN'):
      self.W_gmm_ = tf.get_variable('w_gmm', [args.out_dim, n_direction], initializer=None)
      self.b_gmm = tf.get_variable('b_gmm', [n_direction], initializer=None)

      self.W_state = tf.get_variable('w_state', [args.out_dim, n_state], initializer=None)
      self.b_state = tf.get_variable('b_state', [n_state], initializer=None)

    o_gmm = tf.nn.xw_plus_b(output, self.W_gmm_, self.b_gmm)
    o_state = tf.nn.xw_plus_b(output,self.W_state, self.b_state)

    self.final_state = tf.reshape(self.cell.out,(-1,args.hidden_size + args.out_dim))

    # NB: the below are inner functions, not methods of Model
    def tf_1d_normal(x1, x2, mu1, mu2, s1, s2):
      """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
      norm1 = tf.square(tf.subtract(x1, mu1))
      norm2 = tf.square(tf.subtract(x2, mu2))

      z1 = tf.exp(-1 *tf.div(norm1, 2 * tf.square(s1)))
      z2 = tf.exp(-1 *tf.div(norm2, 2 * tf.square(s2)))

      denom1 = tf.sqrt(2 * np.pi) * s1
      denom2 = tf.sqrt(2 * np.pi) * s2

      result1 = tf.div(z1, denom1)
      result2 = tf.div(z2, denom2)
      return tf.multiply(result1,result2)

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2,
                     z_pen_logits, x1_data, x2_data, pen_data):
      """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
      # This represents the L_R only (i.e. does not include the KL loss term).

      # result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
      #                        z_corr)
      result0 = tf_1d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2)
      epsilon = 1e-6

      # result1 is the loss wrt pen offset (L_s in equation 9 of
      # https://arxiv.org/pdf/1704.03477.pdf)
      Pd = tf.multiply(result0, z_pi)
      Pd = tf.reduce_sum(Pd, 1, keepdims=True)
      logPd = tf.log(Pd + epsilon)  # avoid log(0)

      # fs = 1.0 - pen_data[:, 2]  # use training model for this
      # fs = tf.reshape(fs, [-1, 1])
      # Zero out loss terms beyond N_s, the last actual stroke
      # result1 = tf.multiply(logPd, fs)

      # result2: loss wrt pen state, (L_p in equation 9)
      p = tf.nn.softmax(z_pen_logits)
      logp = tf.log(p + epsilon)
      w = tf.constant([1, 5, 100],dtype=tf.float32)

      result2 = tf.multiply(tf.multiply(w,pen_data),logp)
      result2 = tf.reduce_sum(result2, 1, keep_dims=True)

      result = -tf.reduce_sum(logPd + result2)
      return result

    # below is where we need to do MDN (Mixture Density Network) splitting of
    # distribution params
    def get_mixture_coef(output):
      """Returns the tf slices containing mdn dist params."""
      # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
      z = output
      # z_pen_logits = z[:, 0:3]  # pen states
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2= tf.split(z, 5, 1)

      # process output z's into MDN paramters

      # softmax all the pi's and pen states:
      z_pi = tf.nn.softmax(z_pi)
      # z_pen = tf.nn.softmax(z_pen_logits)

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = tf.exp(z_sigma1)
      z_sigma2 = tf.exp(z_sigma2)
      # z_corr = tf.tanh(z_corr)

      r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2]
      return r

    def get_state_coef(output):
      z = output
      z_pen_logits = z  # pen states
      z_pen = tf.nn.softmax(z_pen_logits)
      return  [z_pen, z_pen_logits]

    out_gmm = get_mixture_coef(o_gmm)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2] = out_gmm

    out_state = get_state_coef(o_state)
    [o_pen, o_pen_logits] = out_state

    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.sigma1 = o_sigma1
    self.sigma2 = o_sigma2
    # o_corr = 0
    self.pen_logits = o_pen_logits
    # pen state probabilities (result of applying softmax to self.pen_logits)
    self.pen = o_pen

    # reshape target model so that it is compatible with prediction shape
    target = tf.reshape(self.output_x, [-1, 5])
    [x1_data, x2_data, cont_data, eos_data, eoc_data] = tf.split(target, 5, 1)
    pen_data = tf.concat([cont_data, eos_data, eoc_data], 1)

    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2,
                            o_pen_logits, x1_data, x2_data, pen_data)

    self.cost = lossfunc

    self.lr = tf.Variable(args.learning_rate, trainable=False)

    optimizer = tf.train.AdamOptimizer(self.lr)

    self.train_op = optimizer.minimize(self.cost, global_step=self.global_step)

    # gradients, variables = zip(*optimizer.compute_gradients(self.cost))
    # gradients = [
    #   None if gradient is None else tf.clip_by_norm(gradient, 5.0)
    #   for gradient in gradients]
    # self.train_op = optimizer.apply_gradients(zip(gradients, variables))


    # gvs = optimizer.compute_gradients(self.cost)
    # g = args.grad_clip
    # capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
    #
    # self.train_op = optimizer.apply_gradients(
    #       capped_gvs, global_step=self.global_step, name='train_step')


def sample(sess, model, seq_len=250, index_char=None, args = ''):
  """Samples a sequence from a pre-trained model."""

  def adjust_temp(pi_pdf, temp):
    pi_pdf = np.log(pi_pdf) / temp
    pi_pdf -= pi_pdf.max()
    pi_pdf = np.exp(pi_pdf)
    pi_pdf /= pi_pdf.sum()
    return pi_pdf

  def get_pi_idx(x, pdf, temp=1.0, greedy=False):
    """Samples from a pdf, optionally greedily."""
    if greedy:
      return np.argmax(pdf)
    pdf = adjust_temp(np.copy(pdf), temp)
    accumulate = 0
    for i in range(0, pdf.size):
      accumulate += pdf[i]
      if accumulate >= x:
        return i
    tf.logging.info('Error with sampling ensemble.')
    return -1

  def random_Pd(vec_mu, vec_sig, vec_pi):
    out = []
    for i in range(len(vec_pi)):
      a = np.random.normal(vec_mu[i], vec_sig[i], 1)*vec_pi[i]
      out.append(a)
    return sum(out)

  prev_x = np.zeros((1, 1, 5), dtype=np.float32)
  prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
  # if z is None:
  #   z = np.random.randn(1, model.hps.z_size)  # not used if unconditional
  #
  prev_state = np.zeros([1, 2*args.hidden_size])

  strokes = np.zeros((seq_len, 5), dtype=np.float32)
  mixture_params = []

  for i in range(seq_len):
    feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state,
          model.index_chars: [index_char]
      }

    params = sess.run([
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2,
        model.pen, model.final_state
    ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_pen, next_state] = params

    if i < 0:
      greedy = False
      temp = 1.0
    else:
      greedy = False
      temp = 1.0

    idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

    idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
    # idx_eos = np.argmax(o_pen[i])
    eos = [0, 0, 0]
    eos[idx_eos] = 1

    # next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
    #                                       o_sigma1[0][idx], o_sigma2[0][idx],
    #                                       np.sqrt(temp), greedy)

    next_x1 = np.random.normal(o_mu1[0][idx], o_sigma1[0][idx])
    next_x2 = np.random.normal(o_mu2[0][idx], o_sigma2[0][idx])

    strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

    params = [
        o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_pen[0]
    ]

    mixture_params.append(params)

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0][0] = np.array(
        [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
    prev_state = next_state

  return strokes, mixture_params