import pandas as pd
import numpy as np
from random import sample
from os.path import basename
with open('driving_log.csv') as f:
    logs = pd.read_csv(f, header=None)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

x_train = logs.ix[:, [0, 1, 2, 3]]
x_train.loc[:, 4] = 0  # flip variable
x_train_flip = x_train.loc[x_train.loc[:, 3] != 0, :]
x_train_flip.loc[:, 4] = 1  # sets to 1, so that process_image flips this image
x_train = x_train.append(x_train_flip)
x_train = x_train.reset_index(drop=True)  # reset indexes

train_rows, val_rows = int(len(x_train) * .8), int(len(x_train) * .9)

x_test = np.array(x_train.loc[(val_rows+1):, :])
x_val = np.array(x_train.loc[(train_rows+1):val_rows, :])
x_train = np.array(x_train.loc[1:train_rows, :])



def remove_zeroes_and_close_to_zero(df, really = True):
    if really:
        # df = df[~((abs(df[:, 3]) < 0.01) & (df[:, 3] != 0))]
        idx = np.random.randint(sum(df[:, 3] == 0), size=int(sum(df[:, 3] == 0) / 2))
        print(len(idx))
        # df = df.delete(idx)
        df = np.delete(df, idx, 0)
        return df


print(x_train.shape)
print(remove_zeroes_and_close_to_zero(x_train).shape)

plt.hist(remove_zeroes_and_close_to_zero(x_train)[:, 3])
plt.show()