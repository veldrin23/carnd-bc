import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import random
import cv2
from sklearn.model_selection import train_test_split
import glob
from os.path import basename
from sklearn.utils import shuffle as shuffledf

def basename2(x):
    return x.split('/')


pd.options.mode.chained_assignment = None

# normalize image
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# flip image
def flip_image(img):
    return cv2.flip(img, flipCode=1)

all_images = glob.glob('F:/images/**/*.jpg',  recursive=True)
image_basenames = [basename(x) for x in all_images]


# read and process image
def read_and_process_img(file_name, flip,  normalize_img=True, grayscale_img=False,
                         remove_top_bottom=True):
    img = mpimg.imread(all_images[image_basenames == basename(file_name)])
    if flip == 1:
        img = flip_image(img)
    return img


# import and shape data from log file

def import_shape_data(logs, add_mirror=True, down_sample_zeroes=True, use_sides=True, side_offset=.08):
    data_in = logs.ix[:, [0, 1, 2, 3]]
    data_in.loc[:, 4] = 0

    # adds extra images with the negative of the driving angle and an indicator to flip the image
    if add_mirror:
        _flip = data_in.loc[data_in.loc[:, 3] != 0, :]
        _flip.loc[:, 3] *= -1
        _flip.loc[:, 4] = 1
        data_in = data_in.append(_flip)

    # down sample zero driving angles
    if down_sample_zeroes:
        zeroes_to_keep = int(sum(data_in.loc[:, 3] == 0) * .5)
        data_in = data_in.loc[data_in.loc[:, 3] == 0, :].sample(zeroes_to_keep)

    # if using all sides
    if use_sides:
        # centre
        data_in_c = data_in.ix[:, [0, 3, 4]]
        data_in_c.columns = ['image_name', 'steering_angle', 'flip']

        # left
        data_in_l = data_in.ix[:, [1, 3, 4]]
        data_in_l.loc[:, 3] += side_offset
        data_in_l.columns = ['image_name', 'steering_angle', 'flip']

        # right
        data_in_r = data_in.ix[:, [2, 3, 4]]
        data_in_r.loc[:, 3] -= side_offset
        data_in_r.columns = ['image_name', 'steering_angle', 'flip']

        data_in = data_in_c.append(data_in_l).append(data_in_r)

    else:
        data_in = data_in.ix[:, [0, 3, 4]]
        data_in.columns = ['image_name', 'steering_angle', 'flip']

    data_in = shuffledf(data_in)

    return data_in

with open('F:/driving_log_udacity.csv') as f:
    _udacity = pd.read_csv(f, header=None, skiprows=1)
    _udacity = import_shape_data(_udacity, down_sample_zeroes=True, add_mirror=True, use_sides=True)


x_train = _udacity