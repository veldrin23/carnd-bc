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

with open('driving_log_step.csv') as f:
    logs = pd.read_csv(f, header=None, skiprows=0)

x_train = logs.ix[:, [0, 1, 2, 3]]

x_train.loc[:, 4] = 0
x_train_flip = x_train.loc[x_train.loc[:, 3] != 0, :]
x_train_flip.loc[:, 3] *= -1
x_train_flip.loc[:, 4] = 1
x_train = x_train.append(x_train_flip)
x_train.loc[:, 5] = 0
x_train.loc[x_train.loc[:, 3] < -0.1, 5] = -1
x_train.loc[x_train.loc[:, 3] > 0.1, 5] = 1
