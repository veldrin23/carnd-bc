import pandas as pd
import numpy as np
from random import sample
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import random
from sklearn.model_selection import train_test_split

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
    data_in = data_in.loc[data_in.loc[:, 5] == 0, ].sample(zeroes_to_keep).\
        append(data_in.loc[data_in.loc[:, 5] < -.01, ]).\
        append(data_in.loc[data_in.loc[:, 5] > .01, ])
    data_in = shuffle(data_in)
    data_in = data_in.reset_index(drop=True)
    data_in.columns = ['centre_image', 'left_image', 'right_image', 'steering_angle', 'flip', 'sharp_turn']
    return data_in

with open('driving_log_all.csv') as f:
    logs = pd.read_csv(f, header=None, skiprows=1)
    full_track = import_shape_data(logs)

with open('driving_log_turn.csv') as f:
    logs = pd.read_csv(f, header=None, skiprows=1)
    turns = import_shape_data(logs)

with open('driving_log_recover.csv') as f:
    logs = pd.read_csv(f, header=None, skiprows=1)
    recover = import_shape_data(logs)




x_train = full_track.append(turns).append(recover)
x_train = x_train.reset_index(drop=True)

train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)
x_test_images, x_test_angles = np.array(x_train.loc[(val_rows+1):,
                                        ['centre_image', 'left_image', 'right_image', 'flip', 'sharp_turn']]), \
                               np.array(x_train.loc[(val_rows+1):,  'steering_angle']).astype(float)

print(x_test_images[...,3][10])

