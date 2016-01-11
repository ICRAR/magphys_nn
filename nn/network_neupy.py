# Magphys neural network
# Date: 10/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Neural network to perform SED fitting
#
# Input: SED inputs (fuv, nuv, u, g etc.)
# Output: Median fit values
import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import random as rand

from neupy.algorithms import GRNN
from neupy.functions import rmsle

import numpy as np
from common.database import get_train_test_data, db_init
import pickle
from keras.utils.visualize_util import to_graph
from pybrain.structure import *
from pybrain.supervised import *
from pybrain.datasets import *

output_names =[
    'ager',
    'tau_V',
    'agem',
    'tlastb',
    'Mstars',
    'xi_Wtot',
    'sfr29',
    'xi_PAHtot',
    'f_muSFH',
    'fb17',
    'fb16',
    'T_CISM',
    'Ldust',
    'mu_parameter',
    'xi_Ctot',
    'f_muIR',
    'fb18',
    'fb19',
    'T_WBC',
    'SFR_0_1Gyr',
    'fb29',
    'sfr17',
    'sfr16',
    'sfr19',
    'sfr18',
    'tau_VISM',
    'sSFR_0_1Gyr',
    'metalicity_Z_Z0',
    'Mdust',
    'xi_MIRtot',
    'tform',
    'gamma']

tmp_file = 'nn_last_tmp_input1.tmp'


def standardise_2Darray(array):

    x_len = len(array)
    y_len = len(array[0])

    means = [0] * len(array[0])
    ranges = [0] * len(array[0])

    for y in range(0, y_len):
        mean = 0
        largest = array[0][y]
        smallest = array[0][y]
        for x in range(0, x_len):
            mean += array[x][y]

            if array[x][y] > largest:
                largest = array[x][y]
            elif array[x][y] < smallest:
                smallest = array[x][y]

        mean /= float(x_len)

        ranged = largest - smallest

        for x in range(0, x_len):
            array[x][y] = (array[x][y] - mean) / float(ranged)

        means[y] = mean
        ranges[y] = ranged

    return means, ranges, array


def normalise_2Darray(array, ignore=0, type=0):
    x_len = len(array)
    y_len = len(array[0])

    minimums = [0] * len(array[0])
    maximums = [0] * len(array[0])

    for y in range(0, y_len):
        minimum = array[y][0]
        maximum = array[y][0]

        for x in range(0, x_len):

            if ignore > 0:
                ignore -= 1
                continue

            if array[x][y] > maximum:
                maximum = array[x][y]
            elif array[x][y] < minimum:
                minimum = array[x][y]

        for x in range(0, x_len):
            if type == 1: # -1 to 1
                array[x][y] = 2*((array[x][y] - minimum) / float(maximum - minimum)) - 1
            else: # 0 to 1
                array[x][y] = ((array[x][y] - minimum) / float(maximum - minimum))

        minimums[y] = minimum
        maximums[y] = maximum

    return minimums, maximums, array


def denormalise_value(value, minimum, maximum):
    return value * (maximum - minimum) + minimum


def check_temp(filename, config):
    if os.path.exists(filename):
        # read in the header
        with open(filename) as f:
            header = pickle.load(f)

            for key in header:
                if header[key] != config[key]:
                    return False

            return True
    else:
        return False


def load_from_file(filename):
    with open(filename) as f:
        header = pickle.load(f)
        test_in = pickle.load(f)
        test_out = pickle.load(f)
        train_in = pickle.load(f)
        train_out = pickle.load(f)
        galaxy_ids = pickle.load(f)


    return test_in, test_out, train_in, train_out, galaxy_ids


def write_file(filename, config, test_in, test_out, train_in, train_out, galaxy_ids):
    with open(filename, 'w') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_in, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_out, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_in, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_out, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(galaxy_ids, f, pickle.HIGHEST_PROTOCOL)


# Inputs: 40 parameters + repeat_redshift * redshift values.
# Repeat redshift can be used to add redshift in to the input layer multiple times.

# Outputs:
# median = 32
# best_fit = 16
# best_fit_model = 4
# best_fit_inputs = 20. That weird line in the fit file that contains different values for the standard inputs.
db_init('sqlite:///Database_run06.db')
train_data = 38000
test_data = 500
run_id = '06'
output_type = 'median'
input_type = 'normal'
repeat_redshift = 1


def run_network(hidden_connections, hidden_layers, loss, single_value=None, normalise=False, input_filter_types=None):

    nn_config_dict = {'test':test_data, 'train':train_data, 'run':run_id, 'input_type': input_type, 'output_type':output_type, 'repeat_redshift':repeat_redshift, 'value':single_value, 'input_filter_types':input_filter_types}

    if check_temp(tmp_file, nn_config_dict):
        print 'Correct temp file exists at {0}, loading from temp'.format(tmp_file)
        test_in, test_out, train_in, train_out, galaxy_ids = load_from_file(tmp_file)
        print 'Done.'
    else:
        print 'No temp file, reading from database.'
        test_in, test_out, train_in, train_out, galaxy_ids = get_train_test_data(test_data, train_data, input_type=input_type,
                                                                                 output_type=output_type,
                                                                                 repeat_redshift=repeat_redshift,
                                                                                 single_value=single_value,
                                                                                 input_filter_types=input_filter_types)

        print 'Done. Writing temp file for next time.'
        write_file(tmp_file, nn_config_dict, test_in, test_out, train_in, train_out, galaxy_ids)
        print 'Done. Temp file written to {0}'.format(tmp_file)

    if normalise:
        print 'Normalising...'
        train_in_min, train_in_max, train_in = normalise_2Darray(train_in)
        train_out_min, train_out_max, train_out = normalise_2Darray(train_out)

        test_in_min, test_in_max, test_in = normalise_2Darray(test_in)
        test_out_min, test_out_max, test_out = normalise_2Darray(test_out)

        print 'Normalising done.'

    print np.shape(train_in)
    print np.shape(train_out)
    print np.shape(test_in)
    print np.shape(test_out)

    print 'Compiling neural network model'

    trained = False
    epoch = 0

    nw = GRNN(std=0.1, verbose=True)

    while not trained:

        nw.train(train_in, train_out)
        result = nw.predict(test_in)
        rmsle(result, test_out)

        print nw.last_error()
        nw.plot_errors()

        epoch += 1

        if epoch == 50:
            trained = True
    print "Compiled."

    # Train the model each generation and show predictions against the validation dataset

    for i in range(0, 30):
        test_to_use = rand.randint(0, test_data - 1)
        ans = nw.predict(np.array([test_in[test_to_use]]))
        #ans = graph.predict({'input':np.array([test_in[test_to_use]]), 'redshift':np.array([redshift_test[test_to_use]])})
        print '\nGalaxy number = {0}\n'.format(galaxy_ids[test_to_use])
        print 'Inputs: {0}\n'.format(test_in[test_to_use])
        print 'Output   Correct\n'
        for a in range(0, len(test_out[test_to_use])):
            print '{0}  =   {1}\n'.format(denormalise_value(ans[0][a], train_out_min[a], train_out_max[a]), denormalise_value(test_out[test_to_use][a], test_out_min[a], test_out_max[a]))
        print '\n\n'

if __name__ == '__main__':
    """hidden_connections = [25, 50, 75, 100]
    hidden_layers = [1, 2, 3, 4]
    values = range(0, 32)
    loss = 'rmse'

    #for value in values:
    for connections in hidden_connections:
        for layer in hidden_layers:
            run_network(connections, layer, loss)"""

    normalise = [True, False]
    filters = ['ir', 'uv', 'optical']
    loss = ['mae', 'mape', 'mse', 'rmse']
    parameters = range(0, 16)

    for parameter in parameters:
        run_network(40, 8, 'mse', parameter, True, ['ir', 'uv', 'optical'])

    exit(0)

    for normalise_type in normalise:
        for loss_type in loss:
            for filter in filters:
                for parameter in parameters:
                    run_network(40, 4, loss_type, normalise=normalise_type, input_filter_types=[filter], single_value=parameter)

print "Done"



