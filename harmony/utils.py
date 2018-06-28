#!/usr/bin/env python

# pylint: disable=R0914

"""Tools for data."""

__author__ = "Matthew Carbone"
__email__ = "x94carbone@gmail.com"

# standardize
def standardize(x_train, x_dev, x_test):
    """Use sklearn's standardizing procedure to rescale the data between
    -1 and 1."""

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)  
    x_train = scaler.transform(x_train) 
    
    x_dev = scaler.transform(x_dev)
    x_test = scaler.transform(x_test)

    return [x_train, x_dev, x_test]


def all_parameters():
    """Read the yaml parameter file hyper.yml and properly distribute the
    hyper parameter parameters (...) into a readable dictionary."""

    import yaml

    protocol = yaml.safe_load(open("hyper.yml"))

    # get the SLURM related variables first
    slurm = protocol['slurm']
    
    # number of runs per hyper parameter data point
    # runtime scales linearly with this variable- meaning if mruns=2, that
    # will be twice as fast as compared with mruns=4
    mruns = protocol['mruns']

    # get the globals for the network
    # can initialize the dictionary here
    hp_dictionary = protocol['globals']
    
    # evaluate the various network layer options, for now there are two:
    # cnn (convolutional neural network)

    # mlp (multi-layer perceptron)
    cnn = protocol['cnn']
    if cnn is None:
        cnn_len = 0
    else:
        cnn_len = len(cnn)
        for layer_number, cnn_layer in enumerate(cnn.values()):
            key_f = 'CNN_f_L' + str(layer_number+1)
            hp_dictionary[key_f] = cnn_layer['f']
            key_k = 'CNN_k_L' + str(layer_number+1)
            hp_dictionary[key_k] = cnn_layer['k']
            key_s = 'CNN_s_L' + str(layer_number+1)
            hp_dictionary[key_s] = cnn_layer['s']
            key_p = 'CNN_p_L' + str(layer_number+1)
            hp_dictionary[key_p] = cnn_layer['p']
            key_a = 'CNN_a_L' + str(layer_number+1)
            hp_dictionary[key_a] = cnn_layer['a']
    hp_dictionary['cnn_len'] = [cnn_len]

    # same thing for the mlp
    mlp = protocol['mlp']
    if mlp is None:
        mlp_len = 0
    else:
        mlp_len = len(mlp)
        for layer_number, mlp_layer in enumerate(mlp.values()):
            key_n = 'MLP_n_L' + str(layer_number)  # MLP is actually 0-indexed
            hp_dictionary[key_n] = mlp_layer['n']
            key_a = 'MLP_a_L' + str(layer_number)
            hp_dictionary[key_a] = mlp_layer['a']
            key_d = 'MLP_d_L' + str(layer_number)
            hp_dictionary[key_d] = mlp_layer['d']
    hp_dictionary['mlp_len'] = [mlp_len]

    return [slurm, mruns, hp_dictionary]



def assert_hyperparameters(hyper):
    """Ensure there are no conflicts within the dictionary, and output
    detailed error messages if there are."""

    errors = []

    # check epochs
    if not 'epochs' in hyper:
        errors.append("epochs must be a key in the hp space")
    elif 0 in hyper['epochs']:
        errors.append("epochs must only have non-zero options")

    # check batch size
    if not 'batch_size' in hyper:
        errors.append("batch_size must be a key in the hp space")
    elif 0 in hyper['batch_size']:
        errors.append("batch_size must only have non-zero options")

    # check fcN0
    if not 'fcN0' in hyper:
        errors.append("fcN0 must be a key in the hp space")
    elif 0 in hyper['fcN0']:
        errors.append("fcN0 must only have non-zero options")

    # end
    if errors != []:
        print('\n--ERROR CHECKS FAILED--\n')
        for i, error in enumerate(errors):
            print("%s: %s" % (str(i+1).zfill(3), error))
        print('\n')
        raise ValueError("Terminating.")

