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
from common.logger import config_logger, add_file_handler_to_root
from network_shared import get_training_data

LOG = config_logger(__name__)
add_file_handler_to_root('nn_run.log')

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

# Inputs: 40 parameters + repeat_redshift * redshift values.
# Repeat redshift can be used to add redshift in to the input layer multiple times.

# Outputs:
# median = 32
# best_fit = 16
# best_fit_model = 4
# best_fit_inputs = 20. That weird line in the fit file that contains different values for the standard inputs.

db_init('sqlite:///Database_run06.db')

config = {'train_data': 38000,
          'test_data': 500,
          'run_id': '06',
          'output_type': 'median',
          'input_type': 'normal',
          'repeat_redshift':  1
          }


def run_network_keras(hidden_connections, hidden_layers, loss, single_output=None, single_input=None, normalise_input=None, normalise_output=None, input_filter_types=None, use_graph=False, optimiser=0):

    config['input_filter_types'] = input_filter_types

    train_data = get_training_data(config, tmp_file, single_output, single_input, normalise_input, normalise_output)

    test_data = config['test_data']
    train_in = np.array(train_data['train_in'])
    train_out = np.array(train_data['train_out'])
    test_in = np.array(train_data['test_in'])
    test_out = np.array(train_data['test_out'])
    redshifts_train = train_data['redshifts_train']
    redshifts_test = train_data['redshifts_test']
    galaxy_ids_train = train_data['galaxy_ids_train']
    galaxy_ids_test = train_data['galaxy_ids_test']
    repeat_redshift = config['repeat_redshift']
    in_normaliser = train_data['in_normaliser']
    out_normaliser = train_data['out_normaliser']

    print np.shape(train_in)
    print np.shape(train_out)
    print np.shape(test_in)
    print np.shape(test_out)

    LOG.info('Compiling neural network model')

    if optimiser == 0:
        optimiser = SGD(lr=0.02, momentum=0.3, decay=0.00001, nesterov=True)
    elif optimiser == 1:
        optimiser = Adagrad(lr=0.01, epsilon=1e-06)
    elif optimiser == 2:
        optimiser = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    elif optimiser == 3:
        optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    if single_input is not None and single_input > 0:
        input_dim = 42
    else:
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

    if single_output is not None and single_output > 0:
        output_dim = 1
    else:
        output_dim = 32

    model = Sequential()
    model.add(Dense(output_dim=hidden_connections, input_dim=output_dim, init='glorot_uniform', activation='tanh'))
    for i in range(0, hidden_layers):
        model.add(Dense(output_dim=hidden_connections, input_dim=hidden_connections, init='glorot_uniform', activation='tanh'))
    model.add(Dense(output_dim=input_dim+repeat_redshift, input_dim=hidden_connections, init='glorot_uniform', activation='linear'))
    model.compile(loss=loss, optimizer=optimiser)

    LOG.info("Compiled.")

    # Train the model each generation and show predictions against the validation dataset
    history = History()
    trained = False
    total_epoch = 0

    history_tracking = []
    history_history = []
    history_seen = []
    LOG.info('nodes_{0}_layers_{1}_filters_{2}_loss_{3}_parameter_{4}_optimiser_{5}_normalise_{6}.txt'.format(hidden_connections, hidden_layers, input_filter_types, loss, 'output_names', optimiser, (normalise_input, normalise_output)))
    while not trained:

        LOG.info('epoch {0}'.format(total_epoch))

        history = model.fit(train_out, train_in, batch_size=1000, nb_epoch=100, validation_split=0.1, show_accuracy=True, verbose=True, callbacks=[history])
        LOG.info('{0}'.format(history.history['loss']))
        total_epoch += 100
        history_seen.append(history.seen)
        history_history.append(history.history)
        history_tracking.append(history.totals)
        if history.totals['loss'] < 0.001 or total_epoch > 999:
            trained = True

        test_to_use = rand.randint(0, test_data - 1)
        if use_graph:
            pass #ans = graph.predict({'input':np.array([test_in[test_to_use]]), 'redshift':np.array([redshift_test[test_to_use]])})
        else:
            ans = model.predict(np.array([test_out[test_to_use]]))

        print '\nGalaxy number = {0}\n'.format(galaxy_ids_test[test_to_use])
        print 'Inputs: {0}\n'.format(test_out[test_to_use])
        print 'Output   Correct\n'
        for a in range(0, len(test_in[test_to_use])):
            if normalise_input:
                #f.write('{0}  =   {1}\n'.format(ans[0][a], test_out[test_to_use][a]))
                print '{0}  =   {1}\n'.format(in_normaliser.denormalise_value(ans[a], a), in_normaliser.denormalise_value(test_in[test_to_use][a], a))
            else:
                print '{0}  =   {1}\n'.format(ans[0][a], test_in[test_to_use][a])
        print '\n\n'

    with open('nodes_{0}_layers_{1}_filters_{2}_loss_{3}_parameter_{4}_optimiser_{5}_normalise_{6}.txt'.format(hidden_connections, hidden_layers, input_filter_types, loss, 'output_names', optimiser, (normalise_input, normalise_output)), 'w') as f:

        for k, v in config.iteritems():
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
            if use_graph:
                pass #ans = graph.predict({'input':np.array([test_in[test_to_use]]), 'redshift':np.array([redshift_test[test_to_use]])})
            else:
                ans = model.predict(np.array([test_out[test_to_use]]))

            f.write('\nGalaxy number = {0}\n'.format(galaxy_ids_test[test_to_use]))
            f.write('Inputs: {0}\n'.format(test_out[test_to_use]))
            f.write('Output   Correct\n')
            for a in range(0, len(test_in[test_to_use])):
                if normalise_input:
                    #f.write('{0}  =   {1}\n'.format(ans[0][a], test_out[test_to_use][a]))
                    f.write('{0}  =   {1}\n'.format(in_normaliser.denormalise_value(ans[a], a), in_normaliser.denormalise_value(test_in[test_to_use][a], a)))
                else:
                    f.write('{0}  =   {1}\n'.format(ans[0][a], test_in[test_to_use][a]))
            f.write('\n\n')

if __name__ == '__main__':
    """
    run_network_keras(60, 4, 'mse', single_output=None, single_input=None, normalise_input=None,
                      normalise_output=None, input_filter_types=None, use_graph=False, optimiser=0)
    """
    normalise = [True, False]
    filters = ['ir', 'uv', 'optical']
    loss = ['mae', 'mse', 'rmse', 'msle', 'squared_hinge', 'hinge', 'binary_crossentropy', 'poisson', 'cosine_proximity']
    loss = ['poisson', 'cosine_proximity']
    parameters = range(0, 15)
    optimisers = range(0, 4)
    hidden_nodes = [10, 20, 40, 80]
    hidden_layers = [1, 2, 3, 4]

    run_network_keras(40, 5, 'mae', single_output=None,
                      single_input=None,
                      normalise_input=None,
                      normalise_output=(0,1),
                      input_filter_types=None,
                      optimiser=0)
    exit(0)

LOG.info("Done")



