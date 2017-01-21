import pandas as pd
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import math
import numpy as np
import json
import tensorflow as tf
from models import *
from random import sample
from sklearn.utils import shuffle
from scipy.misc.pilutil import imresize
from keras.callbacks import ModelCheckpoint
from os.path import basename
from keras.models import model_from_json


pd.options.mode.chained_assignment = None
tf.python.control_flow_ops = tf

#############
# VARIABLES #
#############
nb_epoch = 5
image_rows = int((160 - 50) * .65)
image_columns = int((320 * .65))
batch_size = 350

############
# SETTINGS #
############
fine_tuning = False
grayscale_img = True
use_random_side = False

#########################
# CONDITIONAL VARIABLES #
#########################
if grayscale_img:
    image_channels = 1
else:
    image_channels = 3

if fine_tuning:
    learning_rate = 0.0000001
else:
    learning_rate = 0.0001

#############
# FUNCTIONS #
#############

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
def read_and_process_img(file_name, flip,  normalize_img=True, grayscale_img=grayscale_img,
                         remove_top_bottom=True, resize=True, resize_percentage=.65):

    # img = cv2.imread('F:/IMG/' + basename(file_name), 1)
    img = mpimg.imread(all_images[image_basenames == basename(file_name)])

    if remove_top_bottom:
        img = img[50:, :, :]

    if normalize_img:
        img = normalize(img)

    if flip:
        img = flip_image(img)

    if resize:
        img = imresize(img, resize_percentage, interp='bilinear', mode=None)

    if grayscale_img:
        img = grayscale(img)
        img = img[..., np.newaxis]

    return img


# import and shape data from log file
def import_shape_data(logs, add_mirror=True, down_sample_zeroes=True):
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
        zeroes_to_keep = int(sum(abs(data_in.loc[:, 5])) * .5)  # clumsy, but i think it's kinda cute
        if sum(data_in.loc[:, 5] == 0) > zeroes_to_keep:
            data_in = data_in.loc[data_in.loc[:, 5] == 0, ].sample(zeroes_to_keep).\
                append(data_in.loc[data_in.loc[:, 5] < -.001, ]).\
                append(data_in.loc[data_in.loc[:, 5] > .001, ])
    ###
    data_in = shuffle(data_in)
    data_in = data_in.reset_index(drop=True)
    data_in.columns = ['centre_image', 'left_image', 'right_image', 'steering_angle', 'flip', 'sharp_turn']
    return data_in


with open('F:/driving_log_udacity.csv') as f:
    _udacity = pd.read_csv(f, header=None, skiprows=1)
    _udacity = import_shape_data(_udacity, down_sample_zeroes=True)


with open('F:/driving_log_spoon1.csv') as f:
    _start = pd.read_csv(f, header=None, skiprows=1)
    _start = import_shape_data(_start, down_sample_zeroes=False, add_mirror=False)


x_train = _udacity.append(_start)

print(x_train.shape)
x_train = x_train.reset_index(drop=True)
train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)

x_test_images, x_test_angles = np.array(x_train.loc[(val_rows+1):,
                                        ['centre_image', 'left_image', 'right_image', 'flip']]), \
                               np.array(x_train.loc[(val_rows+1):,  'steering_angle']).astype(float)

x_val_images, x_val_angles = np.array(x_train.loc[(train_rows+1):val_rows,
                                      ['centre_image', 'left_image', 'right_image', 'flip']]),\
                             np.array(x_train.loc[(train_rows+1):val_rows, 'steering_angle']).astype(float)

x_train_images, x_train_angles = np.array(x_train.loc[1:train_rows,
                                          ['centre_image', 'left_image', 'right_image', 'flip']]), \
                                 np.array(x_train.loc[1:train_rows, 'steering_angle']).astype(float)


def get_image(images, angles):
    ii = 0
    while True:

        images_out = np.ndarray(shape=(batch_size, image_rows, image_columns, image_channels), dtype=float)
        angle_out = np.ndarray(shape=batch_size, dtype=float)
        for j in range(batch_size):
            if ii > batch_size:
                images, angles = shuffle(images, angles, random_state=42)
                ii = 0

            if use_random_side:
                random_side = sample(range(3), 1)
                file_name = images[ii, random_side[0]]
                angle_out[j] = angles[ii]
                a = angles[ii]
                if random_side[0] == 1:
                    a += .08
                if random_side[0] == 2:
                    a -= .08
                angle_out[j] = a

            else:
                file_name = images[ii, 0]
                angle_out[j] = angles[ii]

            if images[..., 3][ii] == 1:
                images_out[j] = read_and_process_img(file_name, flip=True)
            else:
                images_out[j] = read_and_process_img(file_name, flip=False)

            ii += 1
        yield images_out, angle_out


def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


if fine_tuning:
    print('back to work...')

    with open('model.json', 'r') as mfile:
        model = model_from_json(json.load(mfile))
    model.load_weights('model.h5')

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse')

else:
    print("starting over again")
    model = nvidia(image_rows, image_columns, image_channels, learning_rate)
    samples_per_epoch = calc_samples_per_epoch(len(x_train_images), batch_size)

    max_q_size = 32
    checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]


history = model.fit_generator(
    get_image(x_train_images, x_train_angles),
    nb_epoch=nb_epoch,
    max_q_size=32,
    samples_per_epoch=calc_samples_per_epoch(len(x_train_images), batch_size),
    validation_data=get_image(x_val_images, x_val_angles),
    nb_val_samples=calc_samples_per_epoch(len(x_val_images), batch_size),
    verbose=1)

score = model.evaluate_generator(
    generator=get_image(x_test_images, x_test_angles),
    val_samples=calc_samples_per_epoch(len(x_test_images), batch_size))

print("Test score {}".format(score))

model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")

# print(angle_array)
# plt.hist(angle_array, 50)
# plt.show()
