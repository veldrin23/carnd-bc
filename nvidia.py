import pandas as pd
import cv2
import matplotlib.image as mpimg
import numpy as np
from random import sample
import math
from random import shuffle
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


fine_tuning = False
tf.python.control_flow_ops = tf

nb_epoch = 2
number_of_rows = int(len([f for f in os.listdir('D:/IMG') if os.path.isfile(os.path.join('D:/IMG', f))])/3)
print(number_of_rows)
image_rows = 78
image_columns = 208
image_channels = 3
batch_size = 150


if fine_tuning:
    learning_rate = 0.0000005
else:
    learning_rate = 0.0000005


# normalize image
def normalize(img):
    return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


# grayscale image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# flip image
def flip_image(img):
    return cv2.flip(img, flipCode=1)


def read_and_process_img(file_name, normalize_img=True, grayscale_img=False, flip=False,
                         remove_top_bottom=True, remove_amount=0.125, resize=True, resize_percentage=.65):

    img = mpimg.imread('D:/IMG/' + basename(file_name), 1)
    if remove_top_bottom:
        img = img[int(img.shape[0]*remove_amount): int(img.shape[0]*(1-remove_amount)), :, :]
    if normalize_img:
        img = normalize(img)
    if grayscale_img:
        img = grayscale(img)
    if flip:
        img = flip_image(img)
    if resize:
        img = imresize(img, resize_percentage, interp='bilinear', mode=None)

    return img


with open('driving_log.csv') as f:
    logs = pd.read_csv(f, header=None)

x_train = logs.ix[:, [0, 1, 2, 3]]
# x_train.loc[:, 4] = 0  # flip variable
x_train_flip = x_train.loc[x_train.loc[:, 3] != 0, :]
x_train_flip.loc[:, 3] *= -1  # sets to 1, so that process_image flips this image
x_train = x_train.append(x_train_flip)
x_train = x_train.reset_index(drop=True)  # reset indexes

train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)

x_test = np.array(x_train.loc[(val_rows+1):, :])
x_val = np.array(x_train.loc[(train_rows+1):val_rows, :])
x_train = np.array(x_train.loc[1:train_rows, :])


def remove_half_of_zeroes(df):
    zeroes_array = np.where(df[:, 3] == 0)[0]
    idx = sample(list(zeroes_array), int(len(zeroes_array)/2))
    df = np.delete(df, idx,0)
    return df


def remove_close_to_zero(df, how_close=0.01):
    df = df[~((abs(df[:, 3]) < how_close) & (df[:, 3] != 0))]
    return df


def reshape_data(x_train, half_zeroes=True, close_to_zero=True):
    if half_zeroes:
        x_train = remove_close_to_zero(x_train)
    if close_to_zero:
        x_train = remove_close_to_zero(x_train)
    return x_train


x_train = reshape_data(x_train)




def get_image(images):
    ii = 0
    while True:

        images_out = np.ndarray(shape=(batch_size, image_rows, image_columns, image_channels), dtype=float)
        angle_out = np.ndarray(shape=batch_size, dtype=float)
        for j in range(batch_size):

            if ii > batch_size:
                shuffle(images)
                ii = 0

            random_side = sample(range(3), 1) # chooses randomly between left, right and centre

            file_name = images[j, random_side]
            angle = images[j, 3]
            #
            # if images[j, 4] == 1:
            #     angle *= 1
            #     images_out[j] = read_and_process_img(file_name[0], flip=True)
            # else:
            images_out[j] = read_and_process_img(file_name[0], flip=False)
            angle_out[j] = angle
            ii += 1
            # print(images_out.shape)
        yield images_out, angle_out


def calc_samples_per_epoch(array_size, batch_size):
    num_batches = array_size / batch_size
    # return value must be a number than can be divided by batch_size
    samples_per_epoch = math.ceil((num_batches / batch_size) * batch_size)
    samples_per_epoch = samples_per_epoch * batch_size
    return samples_per_epoch


def get_model():
    input_shape = (image_rows, image_columns, image_channels)
    model = Sequential()
    model.add(Convolution2D(24,5,5, input_shape=input_shape, subsample = (2,2),
                        border_mode='same',
                        name='conv1', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(36,5,5, subsample = (2,2),
                        border_mode='same',
                        name='conv2', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48,5,5, subsample = (2,2),
                        border_mode='valid',
                        name='conv3', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, subsample = (1,1),
                        border_mode='valid',
                        name='conv4', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64,3,3, subsample = (1,1),
                        border_mode='valid',
                        name='conv5', init='he_normal'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100,name='hidden1', init='he_normal'))
    model.add(ELU())
    model.add(Dense(50,name='hidden2', init='he_normal'))
    model.add(ELU())
    model.add(Dense(10,name='hidden3',init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, name='output', init='he_normal'))
    # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model


if fine_tuning:
    print('back to work...')

    with open('model.json', 'r') as mfile:
        model = model_from_json(json.load(mfile))
    model.load_weights('model.h5')
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

else:
    print("starting over again")
    model = get_model()
    samples_per_epoch = calc_samples_per_epoch(len(x_train), batch_size)

    max_q_size = 32
    checkpoint = ModelCheckpoint("model-{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]


history = model.fit_generator(
    get_image(x_train),
    nb_epoch=nb_epoch,
    max_q_size=10,
    samples_per_epoch=calc_samples_per_epoch(len(x_train), batch_size),
    validation_data=get_image(x_val),
    nb_val_samples=calc_samples_per_epoch(len(x_val), batch_size),
    verbose=1)

score = model.evaluate_generator(
    generator=get_image(x_test),
    val_samples=calc_samples_per_epoch(len(x_test), batch_size))

print("Test score {}".format(score))

model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("./model.h5")
print("Saved model to disk")
