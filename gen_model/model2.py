from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# internal imports

import numpy as np
import tensorflow as tf

import rnn


class Generation_model(object):
    def __init__(self, args, gpu_mode=True, reuse=False, vocabulary=4020):
        with tf.variable_scope('vector_rnn', reuse=reuse):
            self.vocab = vocabulary
            self.build_model(args)

    def build_model(self, args):
        """Define model architecture."""

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, ], name='seq_len')
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, None, 5], name='input')

        self.index_chars = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, ], name='char_index')

        # The target/expected vectors of strokes
        self.output_x = self.input_data[:, 1:args.max_seq_len + 1, :]
        # vectors of strokes to be fed to decoder (same as above, but lagged behind
        # one step to include initial dummy value of (0, 0, 1, 0, 0))
        self.input_x = self.input_data[:, :args.max_seq_len, :]

        # cell_fn = rnn.LSTMCell
        # cell_fn = rnn.GRU

        self.embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab, args.embedding_len], initializer=None)

        #tf.summary.histogram('emmatrix', self.embedding_matrix)

        chars = tf.nn.embedding_lookup(self.embedding_matrix, self.index_chars)

        # if args.is_training:
        #     self.initial_state = tf.nn.tanh(rnn.super_linear(tf.reshape(self.input_x,(-1,args.max_seq_len)),
        #                                                   args.out_dim + args.hidden_size,
        #                                                   init_w='gaussian',
        #                                                   weight_start=0.001,
        #                                                   input_size = None))
        # else:
        self.initial_state = tf.placeholder(shape=[args.max_seq_len, args.hidden_size + args.out_dim], dtype=tf.float32,
                                            name='initial_state')


        # if args.dropout_rate > 0:
        #   cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args.dropout_rate)

        self.cell = rnn.GRU_embedding(x_t=self.input_x,
                                      num_units=args.hidden_size,
                                      c=chars,
                                      state=self.initial_state,
                                      pen_dim=args.pen_dim,
                                      embeding_size=args.embedding_len,
                                      out_dim=args.out_dim)
        # self.cell = rnn.GRU(x_t=self.input_x, hidden_size=args.hidden_size)
        # self.initial_state = self.cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # output = tf.reshape(self.cell.out, (2, -1, args.hidden_size))[1, :, :]
        output = tf.transpose(tf.nn.embedding_lookup(tf.transpose(self.cell.out), tf.range(args.hidden_size,args.hidden_size + args.out_dim)))
        output = tf.reshape(output, [-1, args.out_dim])
        # last_state = tf.reshape(self.cell.out, (2, -1, 500))[0, :, :]

        self.num_mixture = args.num_mixture

        # TODO(deck): Better understand this comment.
        # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
        # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
        # mixture weight/probability (Pi_k)
        n_direction = (args.num_mixture * 6) + 3

        with tf.variable_scope('RNN'):
            self.W_gmm_ = tf.get_variable('w_gmm', [args.out_dim, n_direction], initializer=None)
            self.b_gmm = tf.get_variable('b_gmm', [n_direction], initializer=None)

        output = tf.nn.xw_plus_b(output, self.W_gmm_, self.b_gmm)

        self.final_state = tf.reshape(self.cell.out, (-1, args.hidden_size + args.out_dim))

       # NB: the below are inner functions, not methods of Model
        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
          """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
          norm1 = tf.subtract(x1, mu1)
          norm2 = tf.subtract(x2, mu2)
          s1s2 = tf.multiply(s1, s2)
          # eq 25
          z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
               2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
          neg_rho = 1 - tf.square(rho)
          result = tf.exp(tf.div(-z, 2 * neg_rho))
          denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
          result = tf.div(result, denom)
          return result

        def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                         z_pen_logits, x1_data, x2_data, pen_data):
          """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
          # This represents the L_R only (i.e. does not include the KL loss term).

          result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                                 z_corr)
          epsilon = 1e-6
          # result1 is the loss wrt pen offset (L_s in equation 9 of
          # https://arxiv.org/pdf/1704.03477.pdf)
          result1 = tf.multiply(result0, z_pi)
          result1 = tf.reduce_sum(result1, 1, keep_dims=True)
          result1 = -tf.log(result1 + epsilon)  # avoid log(0)

          fs = 1.0 - pen_data[:, 2]  # use training data for this
          fs = tf.reshape(fs, [-1, 1])
          # Zero out loss terms beyond N_s, the last actual stroke
          result1 = tf.multiply(result1, fs)

          # result2: loss wrt pen state, (L_p in equation 9)
          result2 = tf.nn.softmax_cross_entropy_with_logits(
              labels=pen_data, logits=z_pen_logits)
          result2 = tf.reshape(result2, [-1, 1])

          if not args.is_training:  # eval mode, mask eos columns
              result2 = tf.multiply(result2, fs)

          result = result1 + result2
          return result, result1, result2

        # below is where we need to do MDN (Mixture Density Network) splitting of
        # distribution params
        def get_mixture_coef(output):
          """Returns the tf slices containing mdn dist params."""
          # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
          z = output
          z_pen_logits = z[:, 0:3]  # pen states
          z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

          # process output z's into MDN paramters

          # softmax all the pi's and pen states:
          z_pi = tf.nn.softmax(z_pi)
          z_pen = tf.nn.softmax(z_pen_logits)

          # exponentiate the sigmas and also make corr between -1 and 1.
          z_sigma1 = tf.exp(z_sigma1)
          z_sigma2 = tf.exp(z_sigma2)
          z_corr = tf.tanh(z_corr)

          r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
          return r

        out = get_mixture_coef(output)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.pen_logits = o_pen_logits
        # pen state probabilities (result of applying softmax to self.pen_logits)
        self.pen = o_pen

        # reshape target data so that it is compatible with prediction shape
        target = tf.reshape(self.output_x, [-1, 5])
        [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
        pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

        lossfunc, pd, ps = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                                o_pen_logits, x1_data, x2_data, pen_data)

        self.Pd = tf.reduce_mean(pd)
        self.Ps = tf.reduce_mean(ps)
        self.cost = tf.reduce_mean(lossfunc)
        
        #tf.summary.histogram('loss', self.cost)
        
        #tf.summary.histogram('loss_PD', self.Pd)
        
        #tf.summary.histogram('loss_PS', self.Ps)

        self.lr = tf.Variable(args.learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)

        # self.train_op = optimizer.minimize(self.cost, global_step=self.global_step)

        gvs = optimizer.compute_gradients(self.cost)
        g = args.grad_clip
        capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(
            capped_gvs, global_step=self.global_step, name='train_step')



def sample(sess, model, seq_len=250, temperature=1.0, greedy_mode=False,
           index_char=None, args=[]):
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

  def sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=False):
    if greedy:
      return mu1, mu2
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

  prev_x = np.zeros((1, 1, 5), dtype=np.float32)
  prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke

  prev_state = np.zeros([args.max_seq_len, args.out_dim+args.hidden_size])

  strokes = np.zeros((seq_len, 5), dtype=np.float32)
  mixture_params = []
  greedy = False
  temp = 1.0

  for i in range(seq_len):

    feed = {
          model.input_x: prev_x,
          model.sequence_lengths: [1],
          model.initial_state: prev_state,
          model.index_chars: [index_char]
    }

    params = sess.run([
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
        model.pen, model.final_state
    ], feed)

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

    if i < 0:
      greedy = False
      temp = 1.0
    else:
      greedy = greedy_mode
      temp = temperature

    idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)

    idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)
    eos = [0, 0, 0]
    eos[idx_eos] = 1

    next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx],
                                          o_sigma1[0][idx], o_sigma2[0][idx],
                                          o_corr[0][idx], np.sqrt(temp), greedy)

    strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

    params = [
        o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0],
        o_pen[0]
    ]

    mixture_params.append(params)

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0][0] = np.array(
        [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
    prev_state = next_state

  return strokes, mixture_params


def sample2(sess, model, seq_len=250, index_char=None, args=''):
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
            a = np.random.normal(vec_mu[i], vec_sig[i], 1) * vec_pi[i]
            out.append(a)
        return sum(out)

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    # prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
    # if z is None:
    #   z = np.random.randn(1, model.hps.z_size)  # not used if unconditional
    #
    prev_state = np.zeros([1, 2 * args.hidden_size])

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

        # idx = get_pi_idx(random.random(), o_pi[0], temp, greedy)
        idx = np.argmax(o_pi[0])
        # idx_eos = get_pi_idx(random.random(), o_pen[0], temp, greedy)

        idx_eos = np.argmax(o_pen[0])
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