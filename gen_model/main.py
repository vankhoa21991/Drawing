from random import random
import numpy as np

import tensorflow as tf
from keras.utils import to_categorical

import json, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import *
from data_gen import *
from model import *

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS



def evaluate_model(sess, model, data_set):
  """Returns the average weighted cost, reconstruction cost and KL cost."""
  total_cost = 0.0
  total_r_cost = 0.0
  for batch in range(data_set.num_batches):

    unused_orig_x, x, s,index_chars = data_set.random_batch()

    feed = {model.input_data: x,
            model.sequence_lengths: s,
            model.index_chars: index_chars,
            model.initial_state: np.zeros([args.max_seq_len, 2 * args.hidden_size]),
            }

    (cost, r_cost) = sess.run([model.cost, model.r_cost], feed)
    total_cost += cost
    total_r_cost += r_cost

  total_cost /= (data_set.num_batches)
  total_r_cost /= (data_set.num_batches)
  return (total_cost, total_r_cost)

def train(sess, model, eval_model, train_set, valid_set, test_set,args):
    summary_writer = tf.summary.FileWriter(FLAGS.log_root)

    # Calculate trainable params.
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
    tf.logging.info('Total trainable variables %i.', count_t_vars)
    model_summ = tf.summary.Summary()
    model_summ.value.add(
        tag='Num_Trainable_Params', simple_value=float(count_t_vars))
    summary_writer.add_summary(model_summ, 0)
    summary_writer.flush()

    # setup eval stats
    best_valid_cost = 100000000.0  # set a large init value
    valid_cost = 0.0

    # main train loop

    start = time.time()

    for _ in range(args.num_epochs):

        step = sess.run(model.global_step)

        curr_learning_rate = ((args.learning_rate - args.min_learning_rate) *
                              (args.decay_rate) ** step + args.min_learning_rate)

        _, x, s, index_chars = train_set.random_batch()
        feed = {
            model.input_data: x,
            model.sequence_lengths: s,
            model.lr: curr_learning_rate,
            model.initial_state: np.zeros([args.max_seq_len, 2*args.hidden_size]),
            model.index_chars: index_chars,
        }

        (train_cost, r_cost, _, train_step, _) = sess.run([
            model.cost, model.r_cost, model.final_state,
            model.global_step, model.train_op], feed)

        if step % 20 == 0 and step > 0:
            end = time.time()
            time_taken = end - start

            cost_summ = tf.summary.Summary()
            cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
            reconstr_summ = tf.summary.Summary()
            reconstr_summ.value.add(
                tag='Train_Reconstr_Cost', simple_value=float(r_cost))
            lr_summ = tf.summary.Summary()
            lr_summ.value.add(
                tag='Learning_Rate', simple_value=float(curr_learning_rate))
            time_summ = tf.summary.Summary()
            time_summ.value.add(
                tag='Time_Taken_Train', simple_value=float(time_taken))

            output_format = ('step: %d, lr: %.6f,  cost: %.4f, '
                             'recon: %.4f, train_time_taken: %.4f')
            output_values = (step, curr_learning_rate,  train_cost,
                             r_cost, time_taken)
            output_log = output_format % output_values

            tf.logging.info(output_log)

            summary_writer.add_summary(cost_summ, train_step)
            summary_writer.add_summary(reconstr_summ, train_step)
            summary_writer.add_summary(lr_summ, train_step)
            summary_writer.add_summary(time_summ, train_step)
            summary_writer.flush()
            start = time.time()

        if step % args.save_every == 0 and step > 0:

            (valid_cost, valid_r_cost) = evaluate_model(sess, eval_model, valid_set)

            end = time.time()
            time_taken_valid = end - start
            start = time.time()

            valid_cost_summ = tf.summary.Summary()
            valid_cost_summ.value.add(
                tag='Valid_Cost', simple_value=float(valid_cost))
            valid_reconstr_summ = tf.summary.Summary()
            valid_reconstr_summ.value.add(
                tag='Valid_Reconstr_Cost', simple_value=float(valid_r_cost))
            valid_time_summ = tf.summary.Summary()
            valid_time_summ.value.add(
                tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

            output_format = ('best_valid_cost: %0.4f, valid_cost: %.4f, valid_recon: '
                             '%.4f,  valid_time_taken: %.4f')
            output_values = (min(best_valid_cost, valid_cost), valid_cost,
                             valid_r_cost, time_taken_valid)
            output_log = output_format % output_values

            tf.logging.info(output_log)

            summary_writer.add_summary(valid_cost_summ, train_step)
            summary_writer.add_summary(valid_reconstr_summ, train_step)
            summary_writer.add_summary(valid_time_summ, train_step)
            summary_writer.flush()

            if valid_cost < best_valid_cost:
                best_valid_cost = valid_cost

                save_model(sess, args.model_dir, step)

                end = time.time()
                time_taken_save = end - start
                start = time.time()

                tf.logging.info('time_taken_save %4.4f.', time_taken_save)

                best_valid_cost_summ = tf.summary.Summary()
                best_valid_cost_summ.value.add(
                    tag='Best_Valid_Cost', simple_value=float(best_valid_cost))

                summary_writer.add_summary(best_valid_cost_summ, train_step)
                summary_writer.flush()

                (eval_cost, eval_r_cost) = evaluate_model(sess, eval_model, test_set)

                end = time.time()
                time_taken_eval = end - start
                start = time.time()

                eval_cost_summ = tf.summary.Summary()
                eval_cost_summ.value.add(tag='Eval_Cost', simple_value=float(eval_cost))
                eval_reconstr_summ = tf.summary.Summary()
                eval_reconstr_summ.value.add(
                    tag='Eval_Reconstr_Cost', simple_value=float(eval_r_cost))
                eval_time_summ = tf.summary.Summary()
                eval_time_summ.value.add(
                    tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

                output_format = ('eval_cost: %.4f, eval_recon: %.4f, '
                                 'eval_time_taken: %.4f')
                output_values = (eval_cost, eval_r_cost, time_taken_eval)
                output_log = output_format % output_values

                tf.logging.info(output_log)

                summary_writer.add_summary(eval_cost_summ, train_step)
                summary_writer.add_summary(eval_reconstr_summ, train_step)
                summary_writer.add_summary(eval_time_summ, train_step)
                summary_writer.flush()

def trainer(args):

    # load data
    stroke_train, stroke_val, label_train, label_val, label2char, char2label, max_len = load_data(args.data_dir, args.model_dir)
    vocabulary = len(label2char)

    train_set = DataLoader(stroke_train, label_train, batch_size=args.batch_size,
               max_seq_length=args.max_seq_len, embedding_len = args.embedding_len, trained_embedding= args.trained_embedding, vocabulary = vocabulary)
    valid_set = DataLoader(stroke_val, label_val, batch_size=args.batch_size,
               max_seq_length=args.max_seq_len, embedding_len = args.embedding_len, trained_embedding=args.trained_embedding, vocabulary = vocabulary)
    test_set = valid_set

    reset_graph()
    # load model
    model = Generation_model(args=args,vocabulary=vocabulary)
    eval_model = Generation_model(args=args, reuse=True, vocabulary=vocabulary)

    # start session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # check if resume model
    if args.is_resume:
        load_checkpoint(sess,FLAGS.log_root)

    train(sess, model, eval_model, train_set, valid_set, test_set, args)

def generate(args):
    # load data
    stroke_train, stroke_val, label_train, label_val, label2char, char2label, max_len = load_data(args.data_dir,
                                                                                                  args.model_dir)
    vocabulary = len(label2char)

    test_set = DataLoader(stroke_val, label_val, batch_size=args.batch_size,
                           max_seq_length=args.max_seq_len, embedding_len=args.embedding_len,
                           trained_embedding=args.trained_embedding, vocabulary=vocabulary)


    # construct the sketch-rnn model here:
    reset_graph()

    model = Generation_model(args=args, vocabulary=vocabulary)
    sample_model = Generation_model(args=args, reuse=True, vocabulary=vocabulary)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # loads the weights from checkpoint into our model
    load_checkpoint(sess, FLAGS.log_root)

    _, x, s, index_char = test_set.random_batch()

    sample_strokes, m = sample(sess, sample_model, seq_len=args.max_seq_len, index_char = index_char[0], args = args)

    strokes = to_normal_strokes(sample_strokes)

    draw_strokes(strokes)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # environment
    server = False

    if server == True:
        parser.add_argument('--data_dir', default='/mnt/DATA/lupin/Drawing/recog_model/model/')
        parser.add_argument('--model_dir', default='/mnt/DATA/lupin/Drawing/recog_model/model/')
    else:
        parser.add_argument('--data_dir', default='/home/lupin/Cinnamon/Flaxscanner/Drawing/data/')
        parser.add_argument('--model_dir', default='model/')

    parser.add_argument('--mode', default='trai', type=str)
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--hidden_size', default=500, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--min_learning_rate', default=1e-6, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=int)
    parser.add_argument('--decay_rate', default=0.9999, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--max_seq_len', default=300, type=int)
    parser.add_argument('--num_mixture', default=30, type=int)
    parser.add_argument('--embedding_len', default=500, type=int)
    parser.add_argument('--trained_embedding', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_every', default=20, type=int)
    parser.add_argument('--num_gpu', default='0', type=int)
    parser.add_argument('--is_resume', default=False, type=bool)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)

    tf.app.flags.DEFINE_string(
        'log_root', args.model_dir,
        'Directory to store model checkpoints, tensorboard.')

    if args.mode == 'train':
        trainer(args)

    else:
        generate(args)