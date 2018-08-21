from os.path import isdir
from unittest import TestCase

from PIL import Image

from pycasia import CASIA
import struct
from codecs import decode
from scipy.misc import toimage
import numpy as np
import os
import matplotlib.pyplot as plt
import _pickle as pickle


def my_load_gnt_file(filename):
    """
    Load characters and images from a given GNT file.
    :param filename: The file path to load.
    :return: (image: Pillow.Image.Image, character) tuples
    """

    imgs = []
    lbls = []

    # Thanks to nhatch for the code to read the GNT file, available at https://github.com/nhatch/casia
    with open(filename, "rb") as f:
        while True:
            packed_length = f.read(4)
            if packed_length == b'':
                break

            length = struct.unpack("<I", packed_length)[0]
            raw_label = struct.unpack(">cc", f.read(2))

            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]
            photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))

            # Comes out as a tuple of chars. Need to be combined. Encoded as gb2312, gotta convert to unicode.
            try:
                label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
            except:
                label=''
            # Create an array of bytes for the image, match it to the proper dimensions, and turn it into an image.
            image = toimage(np.array(photo_bytes).reshape(height, width))

            imgs.append(image)
            lbls.append(label)

    return imgs, lbls

def my_load_pot_file(filename):
    """
    Load characters and images from a given GNT file.
    :param filename: The file path to load.
    :return: (image: Pillow.Image.Image, character) tuples
    """

    imgs = []
    lbls = []

    # Thanks to nhatch for the code to read the GNT file, available at https://github.com/nhatch/casia
    with open(filename, "rb") as f:
        COOR = []
        LBLS = []
        while True:
            try:
                packed_length = f.read(2)
                if packed_length == b'':
                    break

                length = struct.unpack("<H", packed_length)[0]
                raw_label = struct.unpack("<cccc", f.read(4))
                try:
                    label = decode(raw_label[1] + raw_label[0], encoding="gb2312")
                except:
                    label = ''
                b = 0

                while b<length-6:

                    stroke_num = struct.unpack("<H", f.read(2))
                    b = b + 2
                    coor_xy = []
                    coor_x = []
                    coor_y = []

                    a = 0
                    c = []
                    cx = []
                    cy = []
                    while a != (-1,-1):
                        a = struct.unpack("<hh", f.read(4))
                        if a == (-1,0):
                            coor_xy.append(c)
                            coor_x.append(cx)
                            coor_y.append(cy)
                            c = []
                            cx = []
                            cy = []
                        else:
                            c.append(a)
                            cx.append(a[0])
                            cy.append(a[1])

                        b = b + 4

                    # fig = plt.figure()
                    # ax = plt.axes()
                    #
                    # for i in range(len(coor_xy)):
                    #     plt.plot(coor_x[i], coor_y[i])
                    # plt.show()
                    COOR.append(coor_xy)
                    LBLS.append(label)
            except Exception as e:
                print(e)
                print('abc')



            # Comes out as a tuple of chars. Need to be combined. Encoded as gb2312, gotta convert to unicode.
            #label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
            # Create an array of bytes for the image, match it to the proper dimensions, and turn it into an image.
            #image = toimage(np.array(photo_bytes).reshape(height, width))

            #imgs.append(image)
            #lbls.append(label)

    return COOR, LBLS

def my_load_ptts_file(filename):
    """
    Load characters and images from a given PTTS file.
    :param filename: The file path to load.
    :return: (image: Pillow.Image.Image, character) tuples
    """

    imgs = []
    lbls = []

    # Thanks to nhatch for the code to read the GNT file, available at https://github.com/nhatch/casia
    with open(filename, "rb") as f:
        COOR = []
        LBLS = []
        while True:
            packed_length = f.read(4)
            if packed_length == b'':
                break

            size_of_header = struct.unpack("<H", packed_length)[0]
            format_code = struct.unpack("<cccc", f.read(8))
            #label = decode(raw_label[1] + raw_label[0], encoding="gb2312")


            b = 0

            while b<length-6:

                stroke_num = struct.unpack("<H", f.read(2))
                b = b + 2
                coor_xy = []
                coor_x = []
                coor_y = []

                a = 0
                c = []
                cx = []
                cy = []
                while a != (-1,-1):
                    a = struct.unpack("<hh", f.read(4))
                    if a == (-1,0):
                        coor_xy.append(c)
                        coor_x.append(cx)
                        coor_y.append(cy)
                        c = []
                        cx = []
                        cy = []
                    else:
                        c.append(a)
                        cx.append(a[0])
                        cy.append(a[1])

                    b = b + 4

                # fig = plt.figure()
                # ax = plt.axes()
                #
                # for i in range(len(coor_xy)):
                #     plt.plot(coor_x[i], coor_y[i])
                # plt.show()
                COOR.append(coor_xy)
                LBLS.append(label)



            # Comes out as a tuple of chars. Need to be combined. Encoded as gb2312, gotta convert to unicode.
            #label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
            # Create an array of bytes for the image, match it to the proper dimensions, and turn it into an image.
            #image = toimage(np.array(photo_bytes).reshape(height, width))

            #imgs.append(image)
            #lbls.append(label)

    return COOR, LBLS

class TestCASIA(TestCase):
    def setUp(self):
        self.casia = CASIA.CASIA()

        # We need at least one dataset to run tests on. Might as well try the smallest one.
        self.casia.get_dataset("HWDB1.1tst_gnt")

    def test_get_all_datasets(self):
        # Make sure that the proper number of datasets are checked.
        self.assertEqual(len(self.casia.datasets), 4)

        # We don't need to test get_dataset because this runs them all
        self.casia.get_all_datasets()
        for dataset in self.casia.datasets:
            dataset_path = self.casia.base_dataset_path + dataset
            self.assertTrue(isdir(dataset_path))

    def test_load_character_images(self):
        for image, character in self.casia.load_character_images():
            self.assertEqual(type(image), Image.Image)
            self.assertEqual(len(character), 1)

def plot_char(char):
    fig = plt.figure()
    ax = plt.axes()

    for i in range(len(char)):
        x,y = zip(*char[i])
        plt.plot(x,y)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

def extract_casia_online(folder):
    list_files = os.listdir(folder)
    # list_files = os.listdir("D:/3_Project/ComputerVision/Cinnamon/Flaxscanner/Dataset/CASIA/Competition_ptts")

    list_labels = []
    list_chars = []
    import numpy.random as random
    for f in list_files:
        try:
            extracted_folder = folder + '/preprocessed/'
            if not os.path.isdir(extracted_folder):
                os.mkdir(extracted_folder)
            char_filepath = extracted_folder + f[:-4] + '_stroke.txt'
            label_filepath = extracted_folder + f[:-4] + '_lbls.txt'
            if not os.path.isfile(char_filepath):
                # imgs, lbls = my_load_gnt_file("D:/3_Project/ComputerVision/Cinnamon/Flaxscanner/Dataset/CASIA/HWDB1.1tst_gnt/" + f)
                chars, lbls = my_load_pot_file(folder + f)
                # imgs, lbls = my_load_ptts_file("D:/3_Project/ComputerVision/Cinnamon/Flaxscanner/Dataset/CASIA/Competition_ptts/" + f)
                list_chars.append(chars)
                list_labels.append(lbls)

                print('Extracting ' + f)
                with open(char_filepath, "wb") as fp:  # Pickling
                    pickle.dump(chars, fp)

                with open(label_filepath, "wb") as fp:  # Pickling
                    pickle.dump(lbls, fp)

            # with open(char_filepath, "rb") as f:
            #     emp = pickle.load(f)

        except:
            print("error at :" + f)

def extract_casia_offline(folder):
    list_files = os.listdir(folder)

    IMGS, LBLS = [], []
    for f in list_files:
        try:
            imgs, lbls = my_load_gnt_file(folder + f)
            IMGS += imgs
            LBLS += lbls

            # for i in range(len(imgs)):
                # imgs[i].save(folder + lbls[i] + "_" + f + "_" + str(i) + ".png")

            # for _ in idx:
            #     img.append(imgs[_])
            #     lbl.append(lbls[_])
            # imgs = img
            # lbls = lbl
            #
            # for i in range(len(imgs)):
            #     if i >= 10:
            #         break
            #     imgs[i].save(folder + lbls[i] + "_" + f + "_" + str(i) + ".png")
            #     if lbls[i] not in list_labels:
            #         list_labels.append(lbls[i])
            #
            # print(f, len(lbls))

        except:
            print("error at :" + f)
    return IMGS, LBLS

def statistic(data_dir):
    list_files = os.listdir(data_dir)
    labels = []
    for f in list_files:
        if f[-9:-4] == '_lbls':
            file = open(data_dir+f, "rb")
            lbls = pickle.load(file)
            labels += lbls
    unique_char = set(labels)
    import collections
    a = collections.Counter(labels)
    print(unique_char)



if __name__ == '__main__':

    # ONLINE
    folder = "D:/3_Project/ComputerVision/Cinnamon/Flaxscanner/Dataset/CASIA/CASIA-OLHWDB1.0/Data/"
    extract_casia_online(folder)

    #data_dir = folder + '/preprocessed/'
    # statistic(data_dir)


    # OFFLINE

    #folder = "D:/3_Project/ComputerVision/Cinnamon/Flaxscanner/Dataset/CASIA/GntData1_0/Data/"
    #IMGS, LBLS = extract_casia_offline(folder)
    #IMGS
