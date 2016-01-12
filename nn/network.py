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

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation
from keras.callbacks import History
from keras.optimizers import *
import numpy as np
from common.database import get_train_test_data, db_init
import pickle
from keras.utils.visualize_util import to_graph
from multiprocessing import Process
from network_pybrain import run_network
from time import sleep

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



def run_network_keras(hidden_connections, hidden_layers, loss, single_value=None, normalise=False, input_filter_types=None, use_graph=False, optimiser=0):

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

    if optimiser == 0:
        optimiser = SGD(lr=0.02, momentum=0.3, decay=0.00001, nesterov=True)
    elif optimiser == 1:
        optimiser = Adagrad(lr=0.01, epsilon=1e-06)
    elif optimiser == 2:
        optimiser = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    elif optimiser == 3:
        optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    first_activation = 'linear'
    if normalise:
        first_activation = 'tanh'

    if input_filter_types is None:
        input_dim = 42
    else:
        input_dim = 0

        if 'optical' in input_filter_types:
            input_dim += 10

        if 'ir' in input_filter_types:
            input_dim += 9

        if 'uv' in input_filter_types:
            input_dim += 2

        input_dim *= 2

    if single_value is not None:
        output_dim = 1
    else:
        output_dim = 15


    if use_graph:
        graph = Graph()

        graph.add_input(name='input', input_shape=(input_dim+repeat_redshift,))
        graph.add_input(name='redshift', input_shape=(1,))
        graph.add_node(Dense(60, input_dim=input_dim+repeat_redshift, activation='tanh'), name='input_layer', input='input')
        graph.add_node(Dense(60, activation='tanh'), name='hidden2', inputs=['input_layer', 'redshift'], merge_mode='concat')
        graph.add_node(Dense(60, activation='tanh'), name='hidden3', input='hidden2')
        graph.add_node(Dense(output_dim, activation='linear'), name='hidden4', input='hidden3')
        graph.add_output(name='output', input='hidden4')

        to_graph(graph).write_svg('graph1.svg')
        graph.compile(loss={'output':'mse'}, optimizer=optimiser)

    else:
        model = Sequential()
        model.add(Dense(output_dim=hidden_connections, input_dim=input_dim+repeat_redshift, init='glorot_uniform', activation=first_activation))
        for i in range(0, hidden_layers):
            model.add(Dense(output_dim=hidden_connections, input_dim=hidden_connections, init='glorot_uniform', activation='tanh'))
        model.add(Dense(output_dim=output_dim, input_dim=hidden_connections, init='glorot_uniform', activation='linear'))
        model.compile(loss=loss, optimizer=optimiser)

    print "Compiled."

    # Train the model each generation and show predictions against the validation dataset
    history = History()
    trained = False
    total_epoch = 0

    history_tracking = []
    history_history = []
    history_seen = []
    while not trained:

        if use_graph:
            pass
            #history = graph.fit({'input':train_in, 'redshift':redshift_train, 'output':train_out}, batch_size=500, nb_epoch=50, validation_split=0.1, verbose=True, callbacks=[history])
        else:
            history = model.fit(train_in, train_out, batch_size=1000, nb_epoch=100, validation_split=0.1, show_accuracy=True, verbose=True, callbacks=[history])

        total_epoch += 100
        history_seen.append(history.seen)
        history_history.append(history.history)
        history_tracking.append(history.totals)
        if history.totals['loss'] < 0.001 or total_epoch > 999:
            trained = True

    with open('filters_{0}_loss_{1}_parameter_{2}_optimiser_{3}.txt'.format(input_filter_types[0], loss, output_names[single_value], optimiser), 'w') as f:

        for k, v in nn_config_dict.iteritems():
            f.write('{0}            {1}\n'.format(k, v))

        i = 0
        f.write('\n\nHistory\n')
        for item in history_history:
            f.write('{0}:   {1}\n'.format(i, item))
            i += 1

        f.write('\n\nSeen\n')
        i = 0
        for item in history_seen:
            f.write('{0}:   {1}\n'.format(i, item))
            i += 1

        f.write('\n\nTotals\n')
        i = 0
        for item in history_tracking:
            f.write('{0}:   {1}\n'.format(i, item))
            i += 1

        for i in range(0, 30):
            test_to_use = rand.randint(0, test_data - 1)
            ans = model.predict(np.array([test_in[test_to_use]]))
            #ans = graph.predict({'input':np.array([test_in[test_to_use]]), 'redshift':np.array([redshift_test[test_to_use]])})
            f.write('\nGalaxy number = {0}\n'.format(galaxy_ids[test_to_use]))
            f.write('Inputs: {0}\n'.format(test_in[test_to_use]))
            f.write('Output   Correct\n')
            for a in range(0, len(test_out[test_to_use])):
                if normalise:
                    #f.write('{0}  =   {1}\n'.format(ans[0][a], test_out[test_to_use][a]))
                    f.write('{0}  =   {1}\n'.format(denormalise_value(ans[0][a], train_out_min[a], train_out_max[a]), denormalise_value(test_out[test_to_use][a], test_out_min[a], test_out_max[a])))
                else:
                    f.write('{0}  =   {1}\n'.format(ans[0][a], test_out[test_to_use][a]))
            f.write('\n\n')

if __name__ == '__main__':

    normalise = [True, False]
    filters = ['ir', 'uv', 'optical']
    loss = ['mae', 'mape', 'mse', 'rmse']
    parameters = range(0, 15)
    optimisers = range(0, 4)

    for filter in filters:
        for parameter in parameters:
            for optimiser in optimisers:
                run_network_keras(60, 4, 'mse', parameter, True, filter, optimiser)


print "Done"



