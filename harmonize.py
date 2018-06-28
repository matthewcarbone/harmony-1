#!/usr/bin/env python

# pylint: disable=R0915

"""Evaluate the results of the hyperparameter runs."""

__author__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"

import pickle
import os

def main():
    """Reads all available results."""

    os_list_directory = os.listdir('train')
    results_list = []

    for index in os_list_directory:
        path = 'train/' + str(index) + '/result.pkl'
        try:
            [params, mean, variance] = pickle.load(open(path, 'rb'))
            results_list.append([index, mean, variance, params])
        except FileNotFoundError:
            pass
    
    sorted_results_list = \
        sorted(results_list, key=lambda k: k[1])
    for entry in sorted_results_list:
        print(entry[0], entry[1], entry[2])


if __name__ == '__main__':
    main()
