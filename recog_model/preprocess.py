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
from utils import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # environment
    server = True

    if server == True:
        parser.add_argument('--data_dir', default='/mnt/DATA/lupin/Flaxscanner/Dataset/CASIA/Drawing/')
        parser.add_argument('--model_dir', default='/mnt/DATA/lupin/Flaxscanner/Models/Drawing/recog_model/')
        parser.add_argument('--sample_dir', default='/mnt/DATA/lupin/Flaxscanner/Drawing/recog_model/samples/')
    else:
        parser.add_argument('--data_dir', default='model/')
        parser.add_argument('--model_dir', default='model/')
    args = parser.parse_args()
    
    stroke_train, stroke_val, stroke_test, label_train, label_val, label_test, label2char, char2label, max_len = load_data(args.data_dir,args.model_dir)
    
    data_filepath = args.data_dir + 'casia_preprocessed_20.npz'     
    
    np.savez_compressed(data_filepath, train=stroke_train, valid=stroke_val, test=stroke_test,label_train=label_train, label_val=label_val,label_test=label_test, label2char=label2char,char2label=char2label,max_len=max_len)
    


