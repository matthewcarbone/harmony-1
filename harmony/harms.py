#!/usr/bin/env python

# pylint: disable=R0913
# pylint: disable=R0914

"""Tools for data."""

__author__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"

from math import floor, log10
import numpy as np

# local imports
from .metrics import metric1

import tensorflow as tf
from keras import backend as K

def multirun(x_train, y_train, x_dev, y_dev, params,
             model_name, directory_index):
    """Executes multiple runs for a given model and computes the average
    quantities corresponding to the output of some metric."""

    import yaml

    # protocol represents the 'global' parameters, but all model fits should
    # be done using params, which has been parsed already and is formatted
    protocol = yaml.safe_load(open("hyper.yml"))
    save_model = protocol['save_model']
    n_runs = protocol['mruns']

    totals = []

    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]

    if params['cnn_len'] != 0:
        x_train = np.expand_dims(x_train, axis=-1)

    order_mag_n_runs = int(floor(log10(n_runs)))


    # for every run
    for i in range(n_runs):

        # print the current progress (output is what is currently
        # being worked on)
        file = open('job_data/%s.txt' % directory_index, 'w')
        file.write("%i/%i\n" % (i+1, n_runs))
        file.close()
        #print("working on multirun %i\n" % (i+1))
        
        #config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        #inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        #device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
        #session = tf.Session(config=config)
        #K.set_session(session)        

        # compile
        model = model_name(n_features, n_classes, params)

        # and execute
        # no validation split, we'll be evaluating separately
        with tf.device('/gpu:0'):
            model.fit(x_train, y_train,
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                      validation_split=0.0, verbose=0)

        # evaluate on the dev set
        current_metric = metric1(model, x_dev, y_dev)
        totals.append(current_metric)

        # save the model
        if save_model:
            model.save('train/%s/model%s.h5' 
                       % (directory_index, str(i).zfill(order_mag_n_runs+1)))

    K.clear_session()
    totals = np.array(totals)

    return [params, np.round(np.mean(totals), 3), np.round(np.var(totals), 3)]
