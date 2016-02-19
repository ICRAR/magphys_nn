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
from keras.layers.core import Dense, Dropout, MaxoutDense, AutoEncoder
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import keras.layers.containers as containers
from keras.callbacks import History
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt
from common.database import db_init
from keras.utils.visualize_util import to_graph
from common.logger import config_logger, add_file_handler_to_root
from network_shared import get_training_data, write_dict, History_Log
from unknown_input import replace_mean, replace_zeros

from network_config import *

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

def random_gen(array):
    """
    Creates an array of random numbers in the same shape as array
    :param array:
    :return:
    """
    return np.random.random(np.shape(array))


def save_differences_graph(differences, filename, epochs_per_entry):
    plt.clf()
    x = range(0, len(differences) * epochs_per_entry, epochs_per_entry)
    y = differences

    print x
    print y

    print np.shape(x)
    print np.shape(y)
    plt.plot(x, y, label='differences', linewidth=2)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('differences')
    plt.xlabel('epochs')

    plt.savefig(filename='{0}.png'.format(filename), format='png')


def save_graph(history, filename, epochs_per_entry):
    """
    Saves a neural network loss graph. With loss and validation loss over epoch
    :param history:
    :param filename:
    :param epochs_per_entry:
    :return:
    """
    plt.clf()
    x = range(0, len(history) * epochs_per_entry, epochs_per_entry)
    y1 = []
    y2 = []

    for item in history:
        y1.append(item['loss'])
        y2.append(item['val_loss'])

    plt.plot(x, y1, label='loss', linewidth=2)
    plt.plot(x, y2, label='val_loss', linewidth=2)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('error rate')
    plt.xlabel('epochs')

    plt.savefig(filename='{0}.png'.format(filename), format='png')

db_init('sqlite:///Database_run06.db')


def run_network_keras(hidden_connections, hidden_layers, loss,
                      single_output=None, single_input=None,
                      normalise_input=None, normalise_output=None,
                      input_filter_types=None,
                      use_graph=False,
                      unknown_input_handler=None):

    config['input_filter_types'] = input_filter_types

    train_data = get_training_data(config,
                                   tmp_file,
                                   single_output,
                                   single_input,
                                   normalise_input,
                                   normalise_output,
                                   unknown_input_handler,
                                   percentile_bin=config['percentile_bin'],
                                   erase_above=config['erase_above'])

    test_data = config['test_data'] # number of test sets

    train_in = train_data['train_in']
    train_out = train_data['train_out']
    test_in = train_data['test_in']
    test_out = train_data['test_out']

    redshifts_train = train_data['redshifts_train']

    galaxy_ids_test = train_data['galaxy_ids_test']

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

    optimiser = SGD(lr=learning_params['learning_rate'], momentum=learning_params['momentum'], decay=learning_params['decay'], nesterov=True)

    input_dim = len(train_in[0])
    try:
        output_dim = len(train_out[0])
    except:
        output_dim = 1

    model = Sequential()

    model.add(Dense(output_dim=hidden_connections, input_dim=input_dim))
    model.add(PReLU())
    model.add(Dense(output_dim=input_dim, input_dim=hidden_connections))

    model.compile(loss='mse', optimizer=RMSprop(lr=0.001), class_mode='binary')

    model.fit(train_in, train_in, 5000, 10000, validation_split=0.3, verbose=True, show_accuracy=True)

    for i in range(0, 30):
        ans = model.predict(np.array([test_in[i]]))

        print 'Test', test_in[i]
        print 'Ans', ans[0]
        print
        print

    exit()

if __name__ == '__main__':
    # Config for the current run
    fitting_params['max_epochs'] = 1000
    fitting_params['epochs_per_fit'] = 100
    fitting_params['batch_size'] = 1000
    net_structure['input_activation'] = 'relu'
    net_structure['hidden_activation'] = 'relu'
    net_structure['output_activation'] = 'linear'
    net_structure['use_dropout'] = True
    config['include_sigma'] = True

    tmp_file = 'nn_last_tmp_input_autoencoder.tmp'

    output_dir = 'test/'

    run_network_keras(22, 0, 'mse', normalise_input=None, normalise_output=None)


LOG.info("Done")



