#!/usr/bin/env python

# pylint: disable=C0411
# pylint: disable=W0611
# pylint: disable=E0401

"""Repository for all kinds of Keras networks."""

__author__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"

import tensorflow
from keras.models import Model
from keras.layers import Dense, Activation, Conv1D, Dropout
from keras.layers import MaxPool1D, Input, BatchNormalization 
from keras.layers import Flatten

def general(n_features, n_classes, params):
    """Standard neural network, allows for a wide variety of parameters.

    :input:
    - n_features/int : the number of features corresponding to the training
      set.
    - n_classes/int : the number of classes ...
    - params/dictionary : contains all paramters used for this run.

    :output:
    - model/keras-type-model : the compiled model which will be used by
      keras.fit later on. 
    """

    if params['cnn_len'] != 0:
        inputs = Input(shape=(n_features, 1), name='Input')
        l_1 = inputs
        for layer in range(params['cnn_len']):
            l_index = str(layer+1)
            l_1 = Conv1D(filters=params['CNN_f_L' + l_index],
                         kernel_size=params['CNN_k_L' + l_index],
                         strides=params['CNN_s_L' + l_index])(l_1)             
            l_1 = BatchNormalization()(l_1)
            if params['CNN_p_L' + l_index] != 0:
                l_1 = MaxPool1D(pool_size=params['CNN_p_L' + l_index])(l_1)
            l_1 = Activation(params['CNN_a_L' + l_index])(l_1)
        l_1 = Flatten()(l_1)

    else:
        inputs = Input(shape=(n_features, ), name='Input')
        l_1 = inputs


    if params['mlp_len'] != 0:
        for mlp_layer in range(params['mlp_len']):
            l_index = str(mlp_layer)
            if params['MLP_n_L' + str(mlp_layer)] != 0:
                l_1 = Dense(params['MLP_n_L' + l_index],
                            activation=params['MLP_a_L' + l_index])(l_1)
                if params['MLP_d_L' + l_index] != 0.0:
                    l_1 = Dropout(params['MLP_d_L' + l_index])(l_1)


    # final activation layer must always be here
    x_1 = Dense(n_classes, activation=params['final_activation'])(l_1)

    model = Model(inputs=inputs, outputs=x_1, name="general")

    model.compile(loss=params['loss'], optimizer=params['optimizer'])

    return model


def main():
    """Do nothing."""
    pass

if __name__ == '__main__':
    main()
