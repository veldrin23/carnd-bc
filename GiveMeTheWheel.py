# import libraries
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import numpy as np
import cv2
import os
from random import sample
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.python.control_flow_ops = tf

##############################
# Variables and other inputs #
##############################

# variables
batch_size = 128
image_dimension_x = 320
image_dimension_y = 160
image_dimension_depth = 1
learning_rate = 0.01
nb_epoch = 10

# options
use_all_three_images = False
grayscale_images = False
normalize_images = True

# and on the 8th day (variables that follow from inputs)

if grayscale_images:
    color_depth = 1
else:
    color_depth = 3
if use_all_three_images:
    image_depth = 3
else:
    image_depth = 1
image_shape = (image_dimension_x, image_dimension_y, color_depth * image_depth)
print(image_shape)

########################
# Additional functions #
########################


# normalize image
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# image read and process function
def read_and_process_img(file_name, normalize_img=normalize_images, grayscale_img=grayscale_images):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if normalize_img:
        img = normalize(img)
    if grayscale_img:
        img = grayscale(img)
    return img



# Generator function
# Thanks to Paul Heraty for this
def img_generator(images, angles):
    ii = 0
    while True:
        images_out = np.ndarray(shape=(batch_size, image_dimension_x, image_dimension_y, image_dimension_depth),
                                dtype=float)  # n*x*y*RGG
        angles_out = np.ndarray(shape=batch_size, dtype=float)
        for j in range(batch_size):
            if ii >= len(images):
                shuffle(images)
                ii = 0
            centre = read_and_process_img(images[ii][0])
            left = read_and_process_img(images[ii][1])
            right = read_and_process_img(images[ii][2])
            angle = angles[ii]
            if use_all_three_images:
                images_out[ii] = np.dstack((centre, left, right))
            else:
                images_out[ii] = centre
            angles_out[ii] = angle
            ii += 1
        yield ({'batchnormalization_input_1': images_out}, {'output': angles_out})


def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch

with open('driving_log.csv') as f:
    logs = pd.read_csv(f)
    nb_images = logs.shape[0]
    images_links = np.ndarray(shape=(nb_images, 3), dtype=object)
    angles = np.ndarray(shape=nb_images, dtype=float)
    i = 0
    for q in logs.iterrows():
        images_links[i, 0] = q[1][0]
        images_links[i, 1] = q[1][1]
        images_links[i, 2] = q[1][2]
        angles[i] = q[1][3]
        i += 1

#############################
# Data extract and handling #
#############################

# create (train, validation) and test data
x_train, x_test, y_train, y_test = train_test_split(images_links, angles, test_size=.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.33, random_state=0)

###############################################
# Model architecture (using VGG architecture) #
###############################################

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=image_shape))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate)
              )
model.summary()

history = model.fit_generator(
    img_generator(x_train, y_train),
    nb_epoch=nb_epoch,
    max_q_size=10,
    samples_per_epoch=calc_samples_per_epoch(len(x_train), batch_size),
    validation_data=img_generator(x_val, y_val),
    nb_val_samples=calc_samples_per_epoch(len(x_val), batch_size),
    verbose=1)

score = model.evaluate_generator(
    generator=img_generator(x_test, y_test),
    val_samples=calc_samples_per_epoch(len(x_test), batch_size))