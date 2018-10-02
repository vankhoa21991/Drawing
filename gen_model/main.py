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
from model2 import *

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS



def evaluate_model(sess, model, data_set):
  """Returns the average weighted cost, reconstruction cost and KL cost."""
  total_cost = 0.0
  total_pd = 0.0
  total_ps = 0.0
  for batch in range(data_set.num_batches):

    unused_orig_x, x, s,index_chars = data_set.random_batch()

    feed = {model.input_data: x,
            model.sequence_lengths: s,
            model.index_chars: index_chars,
            model.initial_state: np.zeros([args.max_seq_len, args.out_dim + args.hidden_size]),
            }


    [cost,pd,ps] = sess.run([model.cost,model.Pd, model.Ps], feed)
    total_cost += cost
    total_pd += pd
    total_ps += ps
    
    
  total_pd /= (data_set.num_batches)
  total_ps /= (data_set.num_batches)
  total_cost /= (data_set.num_batches)

  print('Pd: ' + str(total_pd))
  print('Ps: ' + str(total_ps))

  
  return total_cost

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
    best_valid_cost = 10000000000000.0  # set a large init value
    valid_cost = 0.0

    # main train loop
    embedding_init = sess.run(model.embedding_matrix, feed_dict={})

    start = time.time()

    #train_writer = tf.summary.FileWriter('logs', sess.graph)

    for _ in range(args.num_epochs):

        step = sess.run(model.global_step)

        #merge = tf.summary.merge_all()

        curr_learning_rate = ((args.learning_rate - args.min_learning_rate) *
                              (args.decay_rate) ** step + args.min_learning_rate)

        _, x, s, index_chars = train_set.random_batch()

        feed = {
            model.input_data: x,
            model.sequence_lengths: s,
            model.lr: curr_learning_rate,

            model.initial_state: np.zeros([args.max_seq_len, args.out_dim+args.hidden_size]),
            model.index_chars: index_chars

        }

        (train_cost, _, train_step, _, pd, ps) = sess.run([model.cost,  model.final_state, 
                                                           model.global_step, model.train_op, 
                                                           model.Pd, model.Ps], feed)

        #train_writer.add_summary(summary, step)

        if step % (args.save_every/2)  == 0 and step > 0:

            embedding_after = sess.run(model.embedding_matrix, feed_dict={})
            print('Change in embedding matrix: ' + str(np.sum(abs(embedding_after - embedding_init))))

            end = time.time()
            time_taken = end - start

            cost_summ = tf.summary.Summary()
            cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
            lr_summ = tf.summary.Summary()
            lr_summ.value.add(
                tag='Learning_Rate', simple_value=float(curr_learning_rate))
            time_summ = tf.summary.Summary()
            time_summ.value.add(
                tag='Time_Taken_Train', simple_value=float(time_taken))

            output_format = ('step: %d, lr: %.6f,  cost: %.4f, '
                             'train_time_taken: %.4f')
            output_values = (step, curr_learning_rate,  train_cost,
                            time_taken)
            output_log = output_format % output_values

            tf.logging.info(output_log)

            summary_writer.add_summary(cost_summ, train_step)
            summary_writer.add_summary(lr_summ, train_step)
            summary_writer.add_summary(time_summ, train_step)
            summary_writer.flush()
            start = time.time()

        if step % args.save_every == 0 and step > 0:

            (valid_cost) = evaluate_model(sess, eval_model, valid_set)

            end = time.time()
            time_taken_valid = end - start
            start = time.time()

            valid_cost_summ = tf.summary.Summary()
            valid_cost_summ.value.add(
                tag='Valid_Cost', simple_value=float(valid_cost))

            valid_time_summ = tf.summary.Summary()
            valid_time_summ.value.add(
                tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

            output_format = ('best_valid_cost: %0.4f, valid_cost: %.4f, '
                             ' valid_time_taken: %.4f')
            output_values = (min(best_valid_cost, valid_cost), valid_cost,
                             time_taken_valid)
            output_log = output_format % output_values

            tf.logging.info(output_log)

            summary_writer.add_summary(valid_cost_summ, train_step)
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

                (eval_cost) = evaluate_model(sess, eval_model, test_set)

                end = time.time()
                time_taken_eval = end - start
                start = time.time()

                eval_cost_summ = tf.summary.Summary()
                eval_cost_summ.value.add(tag='Eval_Cost', simple_value=float(eval_cost))
                eval_time_summ = tf.summary.Summary()
                eval_time_summ.value.add(
                    tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

                output_format = ('eval_cost: %.4f, '
                                 'eval_time_taken: %.4f')
                output_values = (eval_cost, time_taken_eval)
                output_log = output_format % output_values

                tf.logging.info(output_log)

                summary_writer.add_summary(eval_cost_summ, train_step)
                summary_writer.add_summary(eval_time_summ, train_step)
                summary_writer.flush()

def trainer(args):

    # load data
    stroke_train, stroke_val, label_train, label_val, label2char, char2label, max_len,_,_ = load_data(args.data_dir, args.model_dir)
    vocabulary = len(label2char)

    train_set = DataLoader(stroke_train, label_train, batch_size=args.batch_size, max_seq_length=args.max_seq_len, embedding_len = args.embedding_len, vocabulary = vocabulary)
    valid_set = DataLoader(stroke_val, label_val, batch_size=args.batch_size, max_seq_length=args.max_seq_len, embedding_len = args.embedding_len, vocabulary = vocabulary)
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
    stroke_train, stroke_val, label_train, label_val, label2char, char2label, max_len, all_strokes, all_lbls = load_data(args.data_dir,
                                                                                                  args.model_dir)
    vocabulary = len(label2char)

    test_set = DataLoader(stroke_val, label_val, batch_size=args.batch_size,
                           max_seq_length=args.max_seq_len, embedding_len=args.embedding_len,
                           vocabulary=vocabulary)

    train_set = DataLoader(stroke_train, label_train, batch_size=args.batch_size,
                          max_seq_length=args.max_seq_len, embedding_len=args.embedding_len,
                          vocabulary=vocabulary)

    data_set = DataLoader(all_strokes, all_lbls, batch_size=args.batch_size,
                           max_seq_length=args.max_seq_len, embedding_len=args.embedding_len,
                           vocabulary=vocabulary)
    # construct the sketch-rnn model here:
    reset_graph()

    model = Generation_model(args=args, vocabulary=vocabulary)
    args.is_training = False
    args.batch_size = 1
    sample_model = Generation_model(args=args, reuse=True, vocabulary=vocabulary)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # print(
    #
    #     "The embedding matrix: " + sess.run(model.embedding_matrix, feed_dict={})
    #
    # )

    # loads the weights from checkpoint into our model
    load_checkpoint(sess, FLAGS.log_root)

    q, x, s, index_char = data_set.random_batch()

    print(label2char.get(index_char[0],None)[0])

    line_rebuild = strokes52lines(x)

    l=0
    for i in range(len(x[0])):
        if x[0][i, 2] > 0:
            l += 1


    plot_char(args.sample_dir,lines2pts(line_rebuild)[0][1:l], label2char.get(index_char[0],None)[0])

    # draw_strokes(to_normal_strokes(x[0]), svg_fpath='sample/origin_' + label2char.get(index_char[0],None)[0] + '.svg')
    # 0: ve 1: nhac len
    # for i in range(len(q)):
    #     char = label2char.get(index_char[i], None)
    #     draw_strokes(to_normal_strokes(q[i]),svg_fpath='sample/origin_'+ char[0] + '.svg')

    sample_strokes, m = sample(sess, sample_model, seq_len=args.max_seq_len, index_char = index_char[0], args = args)


    l=0
    for i in range(len(sample_strokes)):
        if sample_strokes[i, 2] > 0:
            l += 1
        if l == len(sample_strokes):
            l=0

    line_rebuild_gen = strokes52lines([sample_strokes])
    plot_char(args.sample_dir,lines2pts(line_rebuild_gen)[0][1:l], label2char.get(index_char[0],None)[0])
    #print(sample_strokes)
    #strokes = to_normal_strokes(sample_strokes)

    #draw_strokes(strokes)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # environment
    server = True

    if server == True:
        parser.add_argument('--data_dir', default='/mnt/DATA/lupin/Flaxscanner/Dataset/Drawing/')
        parser.add_argument('--sample_dir', default='sample/')
        parser.add_argument('--model_dir', default='/mnt/DATA/lupin/Flaxscanner/Models/Drawing/gen_model2/')
    else:
        parser.add_argument('--data_dir', default='/home/lupin/Cinnamon/Flaxscanner/Dataset/Drawing/')
        parser.add_argument('--sample_dir', default='sample/')
        parser.add_argument('--model_dir', default='/home/lupin/Cinnamon/Flaxscanner/Models/Drawing/gen_model2/')

    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--num_epochs', default= 3000000, type=int)

    parser.add_argument('--hidden_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--min_learning_rate', default=1e-10, type=float)

    parser.add_argument('--grad_clip', default=1.0, type=int)
    parser.add_argument('--decay_rate', default=0.9999, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--max_seq_len', default=60, type=int)
    parser.add_argument('--pen_dim', default=300, type=int)
    parser.add_argument('--out_dim', default=300, type=int)
    parser.add_argument('--num_mixture', default=60, type=int)
    parser.add_argument('--embedding_len', default=500, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--save_every', default=500, type=int)
    parser.add_argument('--num_gpu', default='3', type=int)
    parser.add_argument('--is_resume', default=True, type=bool)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)

    tf.app.flags.DEFINE_string(
        'log_root', args.model_dir,
        'Directory to store model checkpoints, tensorboard.')

    if args.mode == 'train':
        trainer(args)

    else:
        generate(args)