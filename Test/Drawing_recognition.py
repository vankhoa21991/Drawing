import os
from os.path import isdir
from unittest import TestCase

# from PIL import Image

import struct
from codecs import decode
# from scipy.misc import toimage
import numpy as np
import os
import matplotlib.pyplot as plt
import _pickle as pickle
from utils.utils import *



server = False

if server == True:
    data_dir = "/mnt/DATA/lupin/Dataset/CASIA_extracted/"
else:
    data_dir = '/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Online/Data/preprocessed/'


chars_pts_all, lbls_all,_,_ = load_data(data_dir)

# get first 5 people
chars_pts_before_clean = chars_pts_all[:5]
lbls_before_clean = lbls_all[:5]

# index = 30
# for i in range(len(chars)):
#     plot_char(chars[i][index])
# print(lbls[i][index])



chars_pts_after_clean, lbls_after_clean = remove_empty_labels(chars_pts_before_clean, lbls_before_clean)

chars_pts_after_clean = clean_redundant_points(chars_pts_after_clean)

Lines_normalized = []
chars_pts_normalized = []
for i in range(len(chars_pts_after_clean)):

    # points to lines
    lines_before_normalize = pts2lines(chars_pts_after_clean[i])

    # normalize
    lines_after_normalize = normalize(lines_before_normalize)

    #convert back to verify
    chars_pts_normalized.append(lines2pts(lines_after_normalize))

    Lines_normalized.append(lines_after_normalize)

# test normalize
# index = [250, 251, 252]
# i=1
# for inx in index:
#
#     plot_char(chars_pts_after_clean[i][inx])
#     plot_char(chars_pts_normalized[i][inx])
#     print(lbls_after_clean[i][inx])


Lines_input = []
for i in range(len(Lines_normalized)):
    Lines_input.append(extract_line(Lines_normalized[i]))





