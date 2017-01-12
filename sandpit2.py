import pandas as pd
import numpy as np
from random import sample
from os.path import basename
with open('driving_log.csv') as f:
    logs = pd.read_csv(f, header=None)
import matplotlib.image as mpimg


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

#
# for i in range(100):
#     random_side = sample(range(3), 1)
#     # print(basename(x_train[i, random_side][0]))
#     img = mpimg.imread('C:\\New folder\\Img\\' + basename(x_train[i, random_side][0]), 1)
#     print(img.shape)


a = 'D:/IMG/' + basename(logs[1][1])
print(a)
b = mpimg.imread(a)
print(b)
