import pandas as pd
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import math
from sklearn.utils import shuffle

from keras.optimizers import SGD, Adam, RMSprop
from scipy.misc.pilutil import imresize
import tensorflow as tf
import os
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint
from os.path import basename
from keras.models import model_from_json
pd.options.mode.chained_assignment = None
from pprint import pprint
from models import *

tf.python.control_flow_ops = tf

nb_epoch = 5
image_rows = 78
image_columns = 208
batch_size = 65

# what to do.... what to do
fine_tuning = False
grayscale_img = False

if grayscale_img:
    image_channels = 1
else:
    image_channels = 3

if fine_tuning:
    learning_rate = 0.0000025
else:
    learning_rate = 0.00002


# normalize image
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# flip image
def flip_image(img):
    return cv2.flip(img, flipCode=1)


def read_and_process_img(file_name, normalize_img=True, grayscale_img=grayscale_img,
                         remove_top_bottom=True, remove_amount=0.125, resize=True, resize_percentage=.65):
    img = mpimg.imread('F:/CarND2/IMG/' + basename(file_name), 1)
    if remove_top_bottom:
        img = img[int(img.shape[0]*remove_amount): int(img.shape[0]*(1-remove_amount)), :, :]
    if normalize_img:
        img = normalize(img)
    if grayscale_img:
        img = grayscale(img)
    if resize:
        img = imresize(img, resize_percentage, interp='bilinear', mode=None)

    if grayscale_img:
        img = img[..., np.newaxis]
    # print(img.shape)
    return img


def remove_zeroes(df, how_much=.85):
    zeroes_array = np.where(df.loc[:, 3] == 0)[0]
    idx = sample(list(zeroes_array), int(len(zeroes_array)*how_much))
    df = df.drop(df.index[idx])
    return df


def remove_close_to_zero(df, how_close=0.001):
    df = df.loc[(abs(df.loc[:, 3]) == 0) | (abs(df.loc[:, 3]) > how_close), :]
    return df


def reshape_data(x_train, half_zeroes=True, close_to_zero=True):
    if half_zeroes:
        x_train = remove_zeroes(x_train)
    if close_to_zero:
        x_train = remove_close_to_zero(x_train)
    return x_train


with open('driving_log.csv') as f:
    logs = pd.read_csv(f, header=None, skiprows=1)

x_train = logs.ix[:, [0, 1, 2, 3]]
x_train_flip = x_train.loc[x_train.loc[:, 3] != 0, :]
x_train_flip.loc[:, 3] *= -1
x_train = x_train.append(x_train_flip)
x_train = reshape_data(x_train)
x_train = shuffle(x_train)
x_train = x_train.reset_index(drop=True)


train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)
x_test_images, x_test_angles = np.array(x_train.loc[(val_rows+1):, 0:2]), np.array(x_train.loc[(val_rows+1):, 3]).astype(float)
x_val_images, x_val_angles = np.array(x_train.loc[(train_rows+1):val_rows, :]), np.array(x_train.loc[(train_rows+1):val_rows, 3]).astype(float)
x_train_images, x_train_angles = np.array(x_train.loc[1:train_rows, 0:2]), np.array(x_train.loc[1:train_rows, 3]).astype(float)


angle_array = []
def get_image(images, angles):
    ii = 0
    while True:

        images_out = np.ndarray(shape=(batch_size, image_rows, image_columns, image_channels), dtype=float)
        angle_out = np.ndarray(shape=batch_size, dtype=float)
        for j in range(batch_size):

            if ii > batch_size:
                # shuffle(images, angles)
                images, angles = shuffle(images, angles, random_state=42)
                ii = 0

            random_side = sample(range(3), 1)  # chooses randomly between left, right and centre

            file_name = images[j, random_side[0]]
            angle = angles[j]
            images_out[j] = read_and_process_img(file_name)
            angle_out[j] = angle
            angle_array.append(angle)
            ii += 1
            # print(images_out.shape)
        yield images_out, angle_out


def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
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


# plt.hist(angle_array)
# plt.show()
# cv2.waitKey(100)
# cv2.destroyAllWindows()
