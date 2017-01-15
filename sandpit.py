import pandas as pd
import numpy as np
from random import sample
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as df_shuffle

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

with open('driving_log.csv') as f:
    logs = pd.read_csv(f, header=None, skiprows=1)

x_train = logs.ix[:, [0, 1, 2, 3]]
x_train_flip = x_train.loc[x_train.loc[:, 3] != 0, :]
x_train_flip.loc[:, 3] *= -1
x_train = x_train.append(x_train_flip)

x_train = reshape_data(x_train)

x_train = df_shuffle(x_train)
x_train = x_train.reset_index(drop=True)

train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)


# print(train_rows)
print(val_rows)
print(x_train.shape[0])
x_test_images, x_test_angles = np.array(x_train.loc[(val_rows+1):, 0:2]), np.array(x_train.loc[(val_rows+1):, 3]).astype(float)
x_val_images, x_val_angles = np.array(x_train.loc[(train_rows+1):val_rows, :]), np.array(x_train.loc[(train_rows+1):val_rows, 3]).astype(float)
x_train_images, x_train_angles = np.array(x_train.loc[1:train_rows, 0:2]), np.array(x_train.loc[1:train_rows, 3]).astype(float)
x_train = reshape_data(x_train)

print(x_test_images.shape, x_test_angles.shape)
# print(x_test_images.shape, x_test_angles.shape)
# print(x_val_images.shape, x_val_angles.shape)