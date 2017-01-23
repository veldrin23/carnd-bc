import pandas as pd
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Activation
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
pd.options.mode.chained_assignment = None


# from keras.utils.visualize_util import model_to_dot


image_rows = int(110*.85)
image_columns = int(320*.85)
learning_rate = 0.0002
image_channels = 3

def nvidia(image_rows, image_columns, image_channels, learning_rate):
    input_shape = (image_rows, image_columns, image_channels)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - .5, input_shape=input_shape))
    model.add(Convolution2D(24,5,5, subsample = (2,2),
                        border_mode='same',
                        name='conv1', init='he_normal'))
    model.add(Dropout(.5))
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
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100,name='hidden1', init='he_normal'))
    model.add(ELU())
    model.add(Dense(50,name='hidden2', init='he_normal'))
    model.add(ELU())
    model.add(Dense(10,name='hidden3',init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, name='output', init='he_normal'))
    model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
    return model

