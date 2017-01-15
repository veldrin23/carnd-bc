import pandas as pd
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Activation
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
pd.options.mode.chained_assignment = None


def nvidia(image_rows, image_columns, image_channels, learning_rate):
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
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse')
    return model


def model2(image_rows, image_columns, image_channels, learning_rate):
    w_reg = 0.00
    dropout_factor = 0.4
    input_shape1 = (image_rows, image_columns, image_channels)
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

    model = Sequential()
    # CNN Layer 1
    model.add(Lambda(lambda x: x / 128. - 1.,
                     input_shape=input_shape1,
                     output_shape=input_shape1))
    model.add(Convolution2D(nb_filter=num_filters1,
                            nb_row=filter_size1,
                            nb_col=filter_size1,
                            subsample=stride1,
                            border_mode='valid',
                            input_shape=input_shape1,
                            W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # CNN Layer 2
    model.add(Convolution2D(nb_filter=num_filters2,
                            nb_row=filter_size2,
                            nb_col=filter_size2,
                            subsample=stride2,
                            border_mode='valid',
                            W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # CNN Layer 3
    model.add(Convolution2D(nb_filter=num_filters3,
                            nb_row=filter_size3,
                            nb_col=filter_size3,
                            subsample=stride3,
                            border_mode='valid',
                            W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # CNN Layer 4
    model.add(Convolution2D(nb_filter=num_filters4,
                            nb_row=filter_size4,
                            nb_col=filter_size4,
                            subsample=stride4,
                            border_mode='valid',
                            W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # CNN Layer 5
    model.add(Convolution2D(nb_filter=num_filters5,
                            nb_row=filter_size5,
                            nb_col=filter_size5,
                            subsample=stride5,
                            border_mode='valid',
                            W_regularizer=l2(w_reg)))
    model.add(Dropout(dropout_factor))
    model.add(Activation('relu'))
    # Flatten
    model.add(Flatten())
    # FCNN Layer 1
    model.add(Dense(hidden_layers1, input_shape=(2496,), name="hidden1", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # FCNN Layer 2
    model.add(Dense(hidden_layers2, name="hidden2", W_regularizer=l2(w_reg)))
    model.add(Activation('relu'))
    # FCNN Layer 3
    model.add(Dense(1, name="output", W_regularizer=l2(w_reg)))

    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=learning_rate)
                  )
    return model
