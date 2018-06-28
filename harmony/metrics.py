#!/usr/bin/env python

"""Tools for data."""

__author__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"

import numpy as np

def metric1(model, x_test, y_test, how='ok'):
    """Imports a trained model and evaluates based on the metric
    defined here."""

    n_features = x_test.shape[1]

    # analyze results
    def met(y_true, y_pred):
        """Mean absolute error."""

        n_features = len(y_pred)
        sum_ = np.abs(np.round(y_true, 3) - np.round(y_pred, 3))
        return np.sum(sum_)/n_features

    # indices of the various results
    perfect = []
    close = []
    wrong = []

    for i in range(len(x_test)):
        try:
            prediction = model.predict(x_test[i, :].\
                         reshape(1, n_features)).squeeze()
        except ValueError:
            prediction = model.predict(x_test[i, :].\
                         reshape(1, n_features, 1)).squeeze()
        metric = met(y_test[i, :], prediction)
        if metric < 1e-7:
            perfect.append(i)
        elif metric < 0.05:
            close.append(i)
        else:
            wrong.append(i)

    if how == 'perfect':
        return len(perfect)/len(x_test)*100.0
    elif how == 'close':
        return len(close)/len(x_test)*100.0
    elif how == 'wrong':
        return len(wrong)/len(x_test)*100.0
    elif how == 'ok':
        return (len(close) + len(perfect))/len(x_test)*100.0
    else:
        raise ValueError("Unknown evaluation 'how'.")
