import pandas as pd
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import tensorflow as tf
from model_architectures import *
from random import uniform
from sklearn.utils import shuffle as shuffledf # i got heavily confused between skleanr's shuffle and random's shuffle - wasted hours of troubleshooting
from os.path import basename
from keras.models import model_from_json


pd.options.mode.chained_assignment = None
tf.python.control_flow_ops = tf

#############
# VARIABLES #
#############
nb_epoch = 10

image_rows = int(110*.85)
image_columns = int(320*.85)
batch_size = 250

############
# SETTINGS #
############
fine_tuning = False
grayscale_img = False
use_random_side = True


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
    learning_rate = 0.0002


#############
# FUNCTIONS #
#############



def flip_image(img):
    """
    Function to flip image
    :param img: image to flip
    :return: flipped image
    """
    return cv2.flip(img, flipCode=1)


def change_brightness(img):
    """
    Change brightness
    :param img: image array
    :return: brightened image
    """
    change_pct = uniform(0.4, 1.2)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * change_pct
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def read_and_process_img(file_name, flip):
    """
    Function to read in and process images
    :param file_name: name of file that needs to be read and processed
    :param flip: whether imaged should be flipped or not
    :return: image that was cropped, its brightness changed, resized and flipped if required
    """
    img = mpimg.imread('F:/IMG/' + basename(file_name))

    # crop top 50 pixels off
    img = img[50:, :, :]

    # flip image
    if flip == 1:
        img = flip_image(img)

    # Change brightness
    img = change_brightness(img)

    # resize image
    img = cv2.resize(img, (image_columns, image_rows))
    img = img[np.newaxis, ...]

    return img


def import_shape_data(logs, add_mirror=True, down_sample_zeroes=True, use_sides=True, side_offset=.25):
    """
    Function to shape the data from the log files.
    Allows for mirroring, down sampling and to choose to use the left and right sides
    :param logs: Log file created by the simulator
    :param add_mirror: Whether mirror images should be added, default True
    :param down_sample_zeroes: Whether zero values should be down-sampled, default True
    :param use_sides: Whether the side cameras should be used, default True
    :param side_offset: angle offset for the side cameras, default 0.25
    :return: dataset that was augmented and shaped into desired format
    """
    data_in = logs.ix[:, [0, 1, 2, 3]]

    # down sample zero driving angles by removing all the exact 0 angle entries
    if down_sample_zeroes:
        data_in = data_in.loc[data_in.loc[:, 3] != 0, :]

    # using all 3 cameras:
    # if using all sides
    if use_sides:
        # centre
        data_in_c = data_in.ix[:, [0, 3]]
        data_in_c.columns = ['image_name', 'steering_angle']

        # left
        data_in_l = data_in.ix[:, [1, 3]]
        data_in_l.loc[:, 3] = data_in_l.loc[:, 3] - side_offset
        data_in_l.columns = ['image_name', 'steering_angle']

        # right
        data_in_r = data_in.ix[:, [2, 3]]
        data_in_r.loc[:, 3] = data_in_r.loc[:, 3] + side_offset
        data_in_r.columns = ['image_name', 'steering_angle']

        data_in = data_in_c.append(data_in_l).append(data_in_r)

    # if only using centre
    else:
        data_in = data_in.ix[:, [0, 3]]
        data_in.columns = ['image_name', 'steering_angle']

    # add flip variable
    data_in['flip'] = 0
    if add_mirror:
        _flip = data_in[data_in['steering_angle'] != 0]
        _flip['steering_angle'] *= -1
        _flip['flip'] = 1
        data_in = data_in.append(_flip)

    # shuffle df before it exits.
    data_in = shuffledf(data_in)

    return data_in

# sampled_angles = [] # This was used to see if there was an euqal distribution of sampled angles


def get_image(image_list):
    """
    Generator function
    Generator function, saves RAMs by generating data, instead of pushing it all into your memory in one go
    :param image_list:
    :return: images and angles
    """
    ii = 0
    while True:
        images_out = np.ndarray(shape=(batch_size, image_rows, image_columns, image_channels), dtype=float)
        angle_out = np.ndarray(shape=batch_size, dtype=float)

        for j in range(batch_size):
            if ii > batch_size:
                image_list = shuffledf(image_list)
                ii = 0

            angle_out[j] = image_list[ii][1]
            images_out[j] = read_and_process_img(image_list[ii][0], image_list[ii][2])

            ii += 1

            # sampled_angles.append(image_list[ii][1])

        yield images_out, angle_out


def calc_samples_per_epoch(array_size, batch_size):
    """
    Calculates sample per epoc,
    :param array_size: length of the training set (or training, validation set)
    :param batch_size: Batch size
    :return: Sample size for each epoch
    """
    num_batches = array_size / batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


# read in the training data
with open('F:/driving_log_windows.csv') as f:
    _windows = pd.read_csv(f, header=None, skiprows=1)
    _windows = import_shape_data(_windows, down_sample_zeroes=True, add_mirror=True, use_sides=True)

# read in recovery data
with open('F:/driving_log_recover.csv') as f:
    _recover= pd.read_csv(f, header=None, skiprows=1)
    _recover = import_shape_data(_recover, down_sample_zeroes=True, add_mirror=True, use_sides=True)

# combine datasets
x_train = _windows.append(_recover)
# shuffle again
x_train = shuffledf(x_train)
# drop index
x_train.reset_index(drop=True)

# Split data into train, valid and test set
train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)
x_test = np.array(x_train[(val_rows+1):])
x_val = np.array(x_train[(train_rows+1):val_rows])
x_train = np.array(x_train[1:train_rows])


if fine_tuning:
    with open("model.json.save", 'r') as m:
        model = model_from_json(json.load(m))
    weights_file = "model.h5.save"
    model.load_weights(weights_file)

else:
    model = nvidia(image_rows, image_columns, image_channels, learning_rate)


model.summary()

# create history tracker
history = model.fit_generator(
    get_image(x_train),
    nb_epoch=nb_epoch,
    max_q_size=32,
    samples_per_epoch=calc_samples_per_epoch(len(x_train), batch_size),
    validation_data=get_image(x_val),
    nb_val_samples=calc_samples_per_epoch(len(x_val), batch_size),
    verbose=1)

# score checker
score = model.evaluate_generator(
    generator=get_image(x_test),
    val_samples=calc_samples_per_epoch(len(x_test), batch_size))


print("Test score {}".format(score))

# saves model
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")

# plt.hist(sampled_angles, 100)
# plt.show()


