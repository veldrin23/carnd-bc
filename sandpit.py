import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import random
from sklearn.model_selection import train_test_split
import glob
from os.path import basename

pd.options.mode.chained_assignment = None

def remove_zeroes(df, how_much = 1):
    zeroes_array = np.where(df.loc[:, 3] == 0)[0]
    idx = sample(list(zeroes_array), int(len(zeroes_array)*how_much))
    df = df.drop(df.index[idx])
    return df


def remove_close_to_zero(df, how_close=0.01):
    df = df.loc[(abs(df.loc[:, 3]) ==0) | (abs(df.loc[:, 3]) > how_close), :]
    return df


def reshape_data(x_train, half_zeroes=True, close_to_zero=True):
    if half_zeroes:
        x_train = remove_zeroes(x_train)
    if close_to_zero:
        x_train = remove_close_to_zero(x_train)
    return x_train


def import_shape_data(logs):
    data_in = logs.ix[:, [0, 1, 2, 3]]
    data_in.loc[:, 4] = 0
    _flip = data_in.loc[data_in.loc[:, 3] != 0, :]
    _flip.loc[:, 3] *= -1
    _flip.loc[:, 4] = 1
    data_in = data_in.append(_flip)
    data_in.loc[:, 5] = 0
    data_in.loc[data_in.loc[:, 3] < -0.15, 5] = -1
    data_in.loc[data_in.loc[:, 3] > 0.15, 5] = 1
    zeroes_to_keep = int(sum(abs(data_in.loc[:, 5]))*.5)
    if sum(data_in.loc[:, 5] == 0) > zeroes_to_keep:
        data_in = data_in.loc[data_in.loc[:, 5] == 0, ].sample(zeroes_to_keep).\
            append(data_in.loc[data_in.loc[:, 5] < -.01, ]).\
            append(data_in.loc[data_in.loc[:, 5] > .01, ])
    data_in = shuffle(data_in)
    data_in = data_in.reset_index(drop=True)
    data_in.columns = ['centre_image', 'left_image', 'right_image', 'steering_angle', 'flip', 'sharp_turn']
    return data_in


all_images = glob.glob('F:/images/**/*.jpg',  recursive=True)
image_basenames = [basename(x) for x in all_images]
log = 'driving_log_start.csv'

with open('F:/driving_log_start.csv') as f:
    logs = pd.read_csv(f, header=None, skiprows=1)
    _start = import_shape_data(logs)

print(basename(_start['centre_image'][0]))

img = mpimg.imread(all_images[image_basenames == basename(_start['centre_image'][0])])
plt.imshow(img)
plt.show()

