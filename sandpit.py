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

def import_shape_data(logs, add_mirror=True, down_sample_zeroes=False, use_sides=True, side_offset=.3):
    data_in = logs.ix[:, [0, 1, 2, 3]]
    data_in.loc[:, 4] = 0

    # adds extra images with the negative of the driving angle and an indicator to flip the image
    if add_mirror:
        _flip = data_in.loc[data_in.loc[:, 3] != 0, :]
        _flip.loc[:, 3] *= -1
        _flip.loc[:, 4] = 1
        data_in = data_in.append(_flip)

    # downsampe zero driving angles
    data_in.loc[:, 5] = 0
    if down_sample_zeroes:
        ### the next part is to over-sample sharp turning angles and under-sample 0 steering angles
        data_in.loc[data_in.loc[:, 3] < -0.15, 5] = -1
        data_in.loc[data_in.loc[:, 3] > 0.15, 5] = 1
        zeroes_to_keep = int(sum(abs(data_in.loc[:, 5])) * .35)  # clumsy, but i think it's kinda cute
        if sum(data_in.loc[:, 5] == 0) > zeroes_to_keep:
            data_in = data_in.loc[data_in.loc[:, 5] == 0, ].sample(zeroes_to_keep).\
                append(data_in.loc[data_in.loc[:, 5] != 0, ])

    # use all sides
    if use_sides:
        data_in_c = data_in.loc[:, [0, 3, 4, 5]]
        data_in_c.columns = ['image_name', 'steering_angle', 'flip', 'sharp_turn']

        data_in_l = data_in.loc[:, [1, 3, 4, 5]]
        data_in_l.loc[:, 3] += side_offset
        data_in_l.columns = ['image_name', 'steering_angle', 'flip', 'sharp_turn']

        data_in_r = data_in.loc[:, [2, 3, 4, 5]]
        data_in_r.loc[:, 3] -= side_offset
        data_in_r.columns = ['image_name', 'steering_angle', 'flip', 'sharp_turn']

        data_in = data_in_c.append(data_in_l).append(data_in_r)

    else:
        data_in = data_in.ix[:, [0, 3, 4, 5]]
        data_in.columns = ['image_name', 'steering_angle', 'flip', 'sharp_turn']
    data_in = shuffle(data_in)
    data_in = data_in.reset_index(drop=True)
    data_in.columns = ['image_name', 'steering_angle', 'flip', 'sharp_turn']
    return data_in


with open('F:/driving_log_udacity.csv') as f:
    _udacity = pd.read_csv(f, header=None, skiprows=1)
    _udacity = import_shape_data(_udacity, down_sample_zeroes=False, add_mirror=True, use_sides=True)

angle_array = []

x_train = _udacity


x_train = x_train.reset_index(drop=True)
train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)

x_test_images, x_test_angles = np.array(x_train.loc[(val_rows+1):,
                                        ['image_name', 'flip']]), \
                               np.array(x_train.loc[(val_rows+1):,  'steering_angle']).astype(float)

x_val_images, x_val_angles = np.array(x_train.loc[(train_rows+1):val_rows,
                                      ['image_name', 'flip']]),\
                             np.array(x_train.loc[(train_rows+1):val_rows, 'steering_angle']).astype(float)

x_train_images, x_train_angles = np.array(x_train.loc[1:train_rows,
                                          ['image_name', 'flip']]), \
                                 np.array(x_train.loc[1:train_rows, 'steering_angle']).astype(float)

# print(x_train['image_name'][:10])
# print(x_train_images[50:150])

# print(x_train.sample(1))
a = x_train['image_name'][100]


img = read_and_process_img(a, flip=True)
plt.imshow(img)
plt.show()