# Magphys neural network
# Date: 10/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Neural network to perform SED fitting
#
# Input: SED inputs (fuv, nuv, u, g etc.)
# Output: Median fit values
import os, sys, time

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import random as rand

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout
from keras.callbacks import History
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt
from common.database import db_init
from keras.utils.visualize_util import to_graph
from common.logger import config_logger, add_file_handler_to_root
from network_shared import get_training_data, write_dict, History_Log, mean_values
from unknown_input import replace_mean, replace_zeros

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

optimiser_names = [
    'sgd',
    'adagrad',
    'adadelta',
    'adam',
    'rmsprop'
]


def save_graph(history, filename, epochs_per_entry):
    plt.clf()
    x = np.linspace(0, len(history)*epochs_per_entry, len(history))
    y1 = []
    y2 = []

    for item in history:
        y1.append(item['loss'])
        y2.append(item['val_loss'])

    print x
    print y1
    print y2

    plt.plot(x, y1, label='loss', linewidth=2)
    plt.plot(x, y2, label='val_loss', linewidth=2)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('error rate')
    plt.xlabel('epochs')

    plt.savefig(filename='figure {0}.png'.format(filename), format='png')

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

config = None
learning_params = None
fitting_params = None
net_structure = None
tmp_file = None

def run_network_keras(hidden_connections, hidden_layers, loss,
                      single_output=None, single_input=None,
                      normalise_input=None, normalise_output=None,
                      input_filter_types=None,
                      use_graph=False,
                      unknown_input_handler=None,
                      optimiser=0):

    config['input_filter_types'] = input_filter_types

    train_data = get_training_data(config, tmp_file, single_output, single_input, normalise_input, normalise_output, unknown_input_handler)

    test_data = config['test_data'] # number of test sets

    train_in = train_data['train_in']
    train_out = train_data['train_out']
    test_in = train_data['test_in']
    test_out = train_data['test_out']

    redshifts_train = train_data['redshifts_train']
    redshifts_test = train_data['redshifts_test']

    galaxy_ids_train = train_data['galaxy_ids_train']
    galaxy_ids_test = train_data['galaxy_ids_test']

    repeat_redshift = config['repeat_redshift']

    in_normaliser = train_data['in_normaliser']
    out_normaliser = train_data['out_normaliser']

    mean_in = train_data['mean_in']
    mean_out = train_data['mean_out']

    stddev_in = train_data['stddev_in']
    stddev_out = train_data['stddev_out']

    print np.shape(train_in)
    print np.shape(train_out)
    print np.shape(test_in)
    print np.shape(test_out)

    LOG.info('Compiling neural network model')

    if optimiser == 0:
        optimiser = SGD(lr=learning_params['learning_rate'], momentum=learning_params['momentum'], decay=learning_params['decay'], nesterov=True)
    elif optimiser == 1:
        optimiser = Adagrad(lr=0.01, epsilon=1e-06)
    elif optimiser == 2:
        optimiser = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    elif optimiser == 3:
        optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimiser == 4:
        optimiser = RMSprop(lr=0.01, rho=0.9, epsilon=1e-6)

    if single_input is None and single_input > 0:
        input_dim = 1
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

    if config['include_sigma'] is False:
        input_dim /= 2

    if single_output is not None and single_output > 0:
        output_dim = 1
    else:
        output_dim = 32

    if use_graph:
        # Not really used and not up to date.
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
        model.add(Dense(output_dim=hidden_connections, input_dim=input_dim+repeat_redshift, init=net_structure['initialisation_type'], activation=net_structure['input_activation']))
        if net_structure['use_dropout']:
            model.add(Dropout(net_structure['dropout_rate']))
            LOG.info('Using Dropout')
        for i in range(0, hidden_layers):
            model.add(Dense(output_dim=hidden_connections, input_dim=hidden_connections, init=net_structure['initialisation_type'], activation=net_structure['hidden_activation']))
            if net_structure['use_dropout']:
                model.add(Dropout(net_structure['dropout_rate']))
        model.add(Dense(output_dim=output_dim, input_dim=hidden_connections, init=net_structure['initialisation_type'], activation=net_structure['output_activation']))
        model.compile(loss=loss, optimizer=optimiser)

    LOG.info("Compiled.")

    # Train the model each generation and show predictions against the validation dataset
    history = History()
    test = History_Log()
    trained = False
    total_epoch = 0

    history_epoch = []
    history_history = []
    history_totals = []
    test_list = []

    while not trained:

        LOG.info('epoch {0} / {1}'.format(total_epoch, fitting_params['max_epochs']))

        if use_graph:
            history = graph.fit({'input': train_in, 'redshift': redshifts_train, 'output': train_out},
                                batch_size=500,
                                nb_epoch=50,
                                validation_split=0.1,
                                verbose=True, callbacks=[history])
        else:
            history = model.fit(train_in, train_out, batch_size=fitting_params['batch_size'],
                                nb_epoch=fitting_params['epochs_per_fit'],
                                validation_split=fitting_params['validation_split'],
                                show_accuracy=True, verbose=True, callbacks=[history, test])
        LOG.info('{0}'.format(history.history['loss']))
        total_epoch += fitting_params['epochs_per_fit']

        history_epoch.append(history.epoch)
        history_history.append(history.history)
        history_totals.append(history.totals)
        test_list.append(test.epoch_data)

        if history.history['loss'] < 0.001 or total_epoch >= fitting_params['max_epochs']:
            trained = True

    output_file_name = "Keras network run {0}_(replace_zeros)_hidden_{1}.txt".format(int(time.time()), hidden_nodes)
    to_graph(model).write_svg("{0} Graph.svg".format(output_file_name))
    model.save_weights('{0} weights.h5'.format(output_file_name))

    save_graph(test_list, output_file_name, fitting_params['epochs_per_fit'])

    with open(output_file_name, 'w') as f:

        f.write('\n\nNN Configuration (config)\n')
        write_dict(f, config)

        f.write('\n\nNN Configuration (learning params)\n')
        write_dict(f, learning_params)

        f.write('\n\nNN Configuration (net params)\n')
        write_dict(f, net_structure)

        f.write('\n\nNN Configuration (fitting params)\n')
        write_dict(f, fitting_params)

        f.write('\n\nOther params\n')

        f.write('Temp file: {0}\n'.format(tmp_file))
        f.write('Optimiser: {0}\n'.format(optimiser))
        f.write('Single output: {0}\n'.format(single_output))
        f.write('Single input: {0}\n'.format(single_input))
        f.write('Normalise input: {0}\n'.format(normalise_input))
        f.write('Normalise output: {0}\n'.format(normalise_output))
        f.write('Hidden connections: {0}\n'.format(hidden_connections))
        f.write('Hidden layers: {0}\n'.format(hidden_layers))
        if unknown_input_handler:
            f.write('Unknown input handler: {0}\n'.format(unknown_input_handler.__name__))
        else:
            f.write('No unknown input handler')

        i = 0
        f.write('\n\nHistory\n')
        for item in history_history:
            f.write('{0}:   {1}\n'.format(i, item))
            i += 1

        f.write('\n\nEpoch data\n')
        i = 0
        for item in test_list:
            f.write('{0}:   {1}\n'.format(i, item))
            i += 1

        f.write('\n\nTotals\n')
        i = 0
        for item in history_totals:
            f.write('{0}:   {1}\n'.format(i, item))
            i += 1

        f.write('\n\nMean   stddev in\n')
        for i in range(0, len(mean_in)):
            f.write('Input {0}:   {1}   {2}\n'.format(i, mean_in[i], stddev_in[i]))

        f.write('\n\nMean   stddev out\n')
        for i in range(0, len(mean_in)):
            f.write('Input {0}:   {1}   {2}\n'.format(i, mean_out[i], stddev_out[i]))

        for i in range(0, 30):
            test_to_use = rand.randint(0, test_data - 1)
            if use_graph:
                pass #ans = graph.predict({'input':np.array([test_in[test_to_use]]), 'redshift':np.array([redshift_test[test_to_use]])})
            else:
                ans = model.predict(np.array([test_in[test_to_use]]))

            f.write('\nGalaxy number = {0}\n'.format(galaxy_ids_test[test_to_use]))
            f.write('Inputs: {0}\n'.format(test_in[test_to_use]))
            f.write('Output   Correct\n')
            for a in range(0, len(test_out[test_to_use])):
                if normalise_output:
                    #f.write('{0}  =   {1}\n'.format(ans[0][a], test_out[test_to_use][a]))
                    f.write('{0}  =   {1}\n'.format(out_normaliser.denormalise_value(ans[0][a], a), out_normaliser.denormalise_value(test_out[test_to_use][a], a)))
                else:
                    f.write('{0}  =   {1}\n'.format(ans[0][a], test_out[test_to_use][a]))
            f.write('\n\n')

if __name__ == '__main__':

    # Neural network hyper parameters
    replacement_types = [replace_mean, replace_zeros]
    include_sigma = [True, False]
    input_type = ['normal', 'Jy']
    input_filters = ['ir', 'uv', 'optical']
    output_type = ['median', 'best_fit', 'best_fit_model', 'best_fit_inputs']
    output_parameters = range(0, 32)
    normalise = [True, False]

    learning_rate = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    momentum = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    decay = [0.00001, 0.0001, 0.001]
    loss = ['mae', 'mse', 'rmse', 'msle', 'squared_hinge', 'hinge', 'binary_crossentropy', 'poisson', 'cosine_proximity']

    optimisers = range(0, 5)
    hidden_nodes = [50, 80, 120, 160, 200, 300]
    hidden_layers = [1, 2, 3, 4]
    batch_size = [100, 250, 500, 1000, 2000, 5000, 10000]
    epochs_per_fit = [1, 5, 10, 25, 50, 75, 100]
    validation_split = [0.1, 0.2, 0.3]
    max_epochs = [100, 200, 500, 750, 1000]
    activations = ['softmax', 'softplus', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    use_dropout = [True, False]
    dropout_rate = [0.1, 0.25, 0.5, 0.75, 0.9]
    initialisation_type = ['uniform', 'lecun_uniform', 'normal', 'identity', 'orthogonal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']

    config = {'train_data': 200000, 'test_data': 1000,
              'run_id': '06',
              'output_type': 'median',  # median, best_fit, best_fit_model, best_fit_inputs
              'input_type': 'normal',  # normal, Jy
              'include_sigma': False,  # True, False
              'repeat_redshift':  1
              }

    learning_params = {'momentum': 0.3, 'learning_rate': 0.02, 'decay': 0.00001}

    fitting_params = {'batch_size': 1000, 'epochs_per_fit': 100, 'validation_split': 0.3,
                      'max_epochs': 1000
                      }

    net_structure = {'input_activation': 'tanh',  # softmax, softplus, relu, tanh, sigmoid, hard_sigmoid, linear
                     'hidden_activation': 'tanh', 'output_activation': 'tanh',
                     'use_dropout': True, 'dropout_rate': 0.5,  # 0 to 1
                     'initialisation_type': 'glorot_normal'}  # uniform, lecun_uniform, normal, identity, orthogonal, zero, glorot_normal, glorot_uniform, he_normal, he_uniform

    net_structure['input_activation'] = 'relu'
    net_structure['hidden_activation'] = 'relu'
    net_structure['output_activation'] = 'linear'
    fitting_params['max_epochs'] = 1000
    fitting_params['epochs_per_fit'] = 10
    fitting_params['batch_size'] = 10000

    tmp_file = 'nn_last_tmp_input2.tmp'

    for item in hidden_nodes:
            run_network_keras(item, 1, "mse", normalise_input=(0, 1), optimiser=0, unknown_input_handler=replace_zeros)

LOG.info("Done")



