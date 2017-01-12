
# This file generates a Keras mode (model.json) and a corresponding
# weights file (model.h5) which are used to implement behavioral cloning
# for driving a car around a race track. The model takes input frames
# (640x480x3) and labels which contain the steering angle for each frame.
# The model should then be able to predict a steering angle when presented
# which a previously un-seen frame. This can then be used to calculate how
# to steer a car on a track in order to stay on the road

################################################################
# Start by importing the required libraries
################################################################
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from random import shuffle
import scipy.stats as stats
import pylab as pl
import os
import cv2
import csv
import math
import json
from pandas.stats.moments import ewma
from keras.models import model_from_json
import tensorflow as tf
tf.python.control_flow_ops = tf

################################################################
# Define our variables here
################################################################
fine_tune_mode = False

if fine_tune_mode:
    learning_rate = 0.000001
    use_3cams = False
    use_flip = False
else:
    learning_rate = 0.002
    use_3cams = False
    use_flip = True

image_sizeX = 160
image_sizeY = 80
num_channels = 3
n_classes = 1  # This is a regression, not a classification
nb_epoch = 5
batch_size = 100
dropout_factor = 0.4
w_reg = 0.00

input_shape1 = (image_sizeY, image_sizeX, num_channels)
num_filters1 = 24
filter_size1 = 5
stride1 = (2, 2)
num_filters2 = 36
filter_size2 = 5
stride2 = (2, 2)
num_filters3 = 48
filter_size3 = 5
stride3 = (2, 2)
num_filters4 = 64
filter_size4 = 3
stride4 = (1, 1)
num_filters5 = 64
filter_size5 = 3
stride5 = (1, 1)
pool_size = (2, 2)
hidden_layers1 = 100
hidden_layers2 = 50


################################################################
# Define any functions that we need
################################################################

# Read in the image, re-size and flip in necessary
def process_image(filename, flip=0):
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_sizeX, image_sizeY))
    if flip == 1:
        image = cv2.flip(image, 1)
    final_image = image[np.newaxis, ...]
    return final_image


# Calculate the correct number of samples per epoch based on batch size
def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


# Import the training data
# Note: the training image data is stored in the IMG directory, and
# are 640x480 RGB images. Since there will likely be thousands of these
# images, we'll need to use Python generators to access these, thus
# preventing us from running out of memory (which would happen if I
# tried to store the entire set of images in memory as a list

# def get_next_image_angle_pair(image_list):
#     index = 0
#     while 1:
#         final_images = np.ndarray(shape=(batch_size, image_sizeY, image_sizeX, num_channels), dtype=float)
#         final_angles = np.ndarray(shape=(batch_size), dtype=float)
#         for i in range(batch_size):
#             if index >= len(image_list):
#                 index = 0
#                 # Shuffle X_train after every epoch
#                 shuffle(image_list)
#             filename = image_list[index][0]
#             angle = image_list[index][1]
#             flip = image_list[index][2]
#             final_image = process_image(filename, flip)
#             final_angle = np.ndarray(shape=(1), dtype=float)
#             final_angle[0] = angle
#             final_images[i] = final_image
#             final_angles[i] = angle
#             index += 1
#         yield ({'zeropadding2d_input_1': final_images}, {'dense_3': final_angles})
from os.path import basename

def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def read_and_process_img(file_name, normalize_img=True, grayscale_img=False):
    file_name = 'IMG/' + basename(file_name)
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if normalize_img:
        img = normalize(img)
    if grayscale_img:
        img = grayscale(img)
    img = img[np.newaxis, ...]
    return img

image_dimension_x = 320
image_dimension_y = 160
color_depth = 3

# options
use_all_three_images = False
grayscale_images = False
normalize_images = True

def get_next_image_angle_pair(images):
    ii = 0
    while 1:
        images_out = np.ndarray(shape=(batch_size, image_dimension_y, image_dimension_y,3),
                                dtype=float)  # n*x*y*RGG
        angles_out = np.ndarray(shape=batch_size, dtype=float)
        # print(images_out.shape)
        for j in range(batch_size):
            if ii >= len(images):
                shuffle(images)
                ii = 0
            # print(fokken_windows(images[ii][0]))
            centre = read_and_process_img(images[ii][0]).\
                reshape(image_dimension_x, image_dimension_y,  color_depth * 3)
            angle = images[ii][3]

            images_out[ii] = centre
            angles_out[ii] = angle
            # angles_out = angles_out.reshape(batch_size, 1)
            ii += 1
        # yield images_out, angles_out

        yield ({'zeropadding2d_input_1': images_out}, {'dense_3': angles_out})


###############################################
############### START #########################
###############################################

# Start by reading in the .csv file which has the filenames and steering angles
# driving_log_list is a list of lists, where element [x][0] is the image file name
# and element [x][3] is the steering angle
with open('driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    driving_log_list = list(reader)
num_frames = len(driving_log_list)
print("Found {} frames of input data.".format(num_frames))

# Process this list so that we end up with training images and labels
if use_3cams:
    X_train = [("", 0.0, 0) for x in range(num_frames * 3)]
    print(len(X_train))
    for i in range(num_frames):
        X_train[i * 3] = (driving_log_list[i][0].lstrip(),  # center image
                          float(driving_log_list[i][3]),  # center angle
                          0)  # dont flip
        X_train[(i * 3) + 1] = (driving_log_list[i][1].lstrip(),  # left image
                                float(driving_log_list[i][3]) + 0.08,  # left angle
                                0)  # dont flip
        X_train[(i * 3) + 2] = (driving_log_list[i][2].lstrip(),  # right image
                                float(driving_log_list[i][3]) - 0.08,  # right angle
                                0)  # dont flip
else:
    X_train = [("", 0.0, 0) for x in range(num_frames)]
    print(len(X_train))
    for i in range(num_frames):
        # print(i)
        X_train[i] = (driving_log_list[i][0].lstrip(),  # center image
                      float(driving_log_list[i][3]),  # center angle
                      0)  # dont flip

# Update num_frames as needed
num_frames = len(X_train)

# Also, in order to generate more samples, lets add entries twice for
# entries that have non-zero angles, and add a flip switch. Then when
# we are reading these, we will flip the image horizontally and
# negate the angles
if use_flip:
    for i in range(num_frames):
        if X_train[i][1] != 0.0:
            X_train.append([X_train[i][0], -1.0 * X_train[i][1], 1])  # flip flag

num_frames = len(X_train)
print("After list pre-processing, now have {} frames".format(num_frames))

# Split some of the training data into a validation dataset.
# First lets shuffle the dataset, as we added lots of non-zero elements to the end
shuffle(X_train)
num_train_elements = int((num_frames / 4.) * 3.)
num_valid_elements = int(((num_frames / 4.) * 1.) / 2.)
X_valid = X_train[num_train_elements:num_train_elements + num_valid_elements]
X_test = X_train[num_train_elements + num_valid_elements:]
X_train = X_train[:num_train_elements]
print("X_train has {} elements.".format(len(X_train)))
print("X_valid has {} elements.".format(len(X_valid)))
print("X_test has {} elements.".format(len(X_test)))

################################################################
# Load the existing model  & weights if we are fine tuning
################################################################
if fine_tune_mode:
    print("**********************************")
    print("*** Running in FINE-TUNE mode! ***")
    print("**********************************")
    with open("model.json.save", 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = "model.h5.save"
    model.load_weights(weights_file)
else:
    ################################################################
    # Otherwise build a new CNN Network with Keras
    ################################################################

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape1))
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
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax', name='output'))

# model.summary()

################################################################
# Train the network using generators
################################################################
print("Number of Epochs : {}".format(nb_epoch))
print("  Batch Size : {}".format(batch_size))
print("  Training batches : {} ".format(calc_samples_per_epoch(len(X_train), batch_size)))
print("  Validation batches : {} ".format(calc_samples_per_epoch(len(X_valid), batch_size)))

if fine_tune_mode:
    print("*** Fine-tuning model with learning rate {} ***".format(learning_rate))
else:
    print("*** Compiling new model wth learning rate {} ***".format(learning_rate))

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=learning_rate)
              )

history = model.fit_generator(
    get_next_image_angle_pair(X_train),  # The generator to return batches to train on
    nb_epoch=nb_epoch,  # The number of epochs we will run for
    max_q_size=10,  # Max generator items that are queued and ready
    samples_per_epoch=calc_samples_per_epoch(len(X_train), batch_size),
    validation_data=get_next_image_angle_pair(X_valid),  # validation data generator
    nb_val_samples=calc_samples_per_epoch(len(X_valid), batch_size),
    verbose=1)

# Evaluate the accuracy of the model using the test set
score = model.evaluate_generator(
    generator=get_next_image_angle_pair(X_test),  # validation data generator
    val_samples=calc_samples_per_epoch(len(X_test), batch_size),  # How many batches to run in one epoch
)
print("Test score {}".format(score))

################################################################
# Save the model and weights
################################################################
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")
