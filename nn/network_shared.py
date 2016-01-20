# Shared neural network functions

import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import numpy as np
from common.database import get_train_test_data
from math import sqrt
import pickle
from keras.callbacks import Callback
from common.logger import config_logger

LOG = config_logger(__name__)


class ArrayNormaliser(object):
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

        self.dataset_mins = None
        self.dataset_maxs = None

    def _scale_array_to(self, array, lower, upper):
        x_len = len(array)
        y_len = len(array[0])

        minimums = [0] * len(array[0])
        maximums = [0] * len(array[0])

        for y in range(0, y_len):
            minimum = array[0][y]
            maximum = array[0][y]

            for x in range(0, x_len):

                if array[x][y] > maximum:
                    maximum = array[x][y]
                elif array[x][y] < minimum:
                    minimum = array[x][y]

            for x in range(0, x_len):
                if type(upper) == list and type(lower) == list:
                    array[x][y] = ((array[x][y] - minimum) / float(maximum - minimum)) * (upper[y] - lower[y]) + lower[y]
                else:
                    array[x][y] = ((array[x][y] - minimum) / float(maximum - minimum)) * (upper - lower) + lower

            minimums[y] = minimum
            maximums[y] = maximum

        self.dataset_maxs = maximums
        self.dataset_mins = minimums

        return array

    def normalise_array(self, array):
        return self._scale_array_to(array, self.min, self.max)

    def denormalise_array(self, array):
        return self._scale_array_to(array, self.dataset_mins, self.dataset_maxs)

    def denormalise_value(self, value, array_index):
        """
        :param value the value to scale
        :param the y array index corresponds to the type of value that should be scaled (used to get min and max for that y set)
        """

        if value > self.max or value < self.min:
            return None

        return ((value - self.min) / (self.max - self.min)) * (self.dataset_maxs[array_index]  - self.dataset_mins[array_index]) + self.dataset_mins[array_index]


class History_Log(Callback):

    def __init__(self):
        super(Callback, self).__init__()
        self.last_batch = None
        self.epoch_data = None

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.last_batch = logs['loss']

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_data = {'loss': self.last_batch, 'val_loss': logs['val_loss']}


def mean_values(array):
    means = []
    for item in range(0, len(array[0])):
        means.append(np.mean(array[:,item]))

    return means


def std_dev(array):
    std = []
    for item in range(0, len(array[0])):
        std.append(np.std(array[:, item]))

    return std


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


def split_data(array, split_point):
    # Everything up until split point       Everything after split point
    return array[:split_point], array[split_point:]


def load_from_file(filename):
    with open(filename) as f:
        header = pickle.load(f)
        all_in = pickle.load(f)
        all_out = pickle.load(f)
        redshifts = pickle.load(f)
        galaxy_ids = pickle.load(f)

    return all_in, all_out, redshifts, galaxy_ids


def write_file(filename, config, all_in, all_out, redshifts, galaxy_ids):
    with open(filename, 'w') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_in, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_out, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(redshifts, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(galaxy_ids, f, pickle.HIGHEST_PROTOCOL)


def get_single_dimension(array, dimension):
    # Get only a single value from the output
    whole_set = []

    for item in array:
        whole_set.append([item[dimension]])

    return np.array(whole_set)


def get_training_data(config, tmp_file, single_output=None, single_input=None,
                      normalise_input=None, normalise_output=None,
                      unknown_input_handler=None):

    nn_config_dict = {'test':config['test_data'],
                      'train':config['train_data'],
                      'run':config['run_id'],
                      'input_type': config['input_type'],
                      'output_type':config['output_type'],
                      'repeat_redshift':config['repeat_redshift'],
                      'input_filter_types':config['input_filter_types'],
                      'include_sigma': config['include_sigma'],
                      'unknown_input_handler': unknown_input_handler}

    if check_temp(tmp_file, nn_config_dict):
        LOG.info('Correct temp file exists at {0}, loading from temp'.format(tmp_file))
        all_in, all_out, redshifts, galaxy_ids = load_from_file(tmp_file)
        LOG.info('Done.')
    else:
        LOG.info('No temp file, reading from database.')
        all_in, all_out, redshifts, galaxy_ids = get_train_test_data(nn_config_dict['test'],
                                  nn_config_dict['train'],
                                  input_type=nn_config_dict['input_type'],
                                  output_type=nn_config_dict['output_type'],
                                  include_sigma=nn_config_dict['include_sigma'],
                                  repeat_redshift=nn_config_dict['repeat_redshift'],
                                  input_filter_types=nn_config_dict['input_filter_types'],
                                  unknown_input_handler=unknown_input_handler)

        LOG.info('Done. Writing temp file for next time.')
        write_file(tmp_file, nn_config_dict, all_in, all_out, redshifts, galaxy_ids)
        LOG.info('Done. Temp file written to {0}'.format(tmp_file))

    print 'In shape'
    print np.shape(all_in)
    print 'Out shape'
    print np.shape(all_out)
    print 'Gal id'
    print np.shape(galaxy_ids)
    print 'Redshift'
    print np.shape(redshifts)

    print '\n\n\nFirst 5'
    for i in range(0, 5):
        print all_in[i]
        print all_out[i]
        print galaxy_ids[i]
        print redshifts[i]

    if single_output is not None and single_output > 0:
        # Get only a single value from the output
        all_out = get_single_dimension(all_out, single_output)

    if single_input is not None and single_input > 0:
        # Get only a single value from the output
        all_in = get_single_dimension(all_in, single_input)

    all_in_normaliser = None
    all_out_normaliser = None

    if normalise_input is not None:
        LOG.info('Normalising input...')
        all_in_normaliser = ArrayNormaliser(normalise_input[0], normalise_input[1])

        all_in = all_in_normaliser.normalise_array(all_in)

        LOG.info('Normalising done.')

    if normalise_output is not None:
        LOG.info('Normalising output...')
        all_out_normaliser = ArrayNormaliser(normalise_output[0], normalise_output[1])

        all_out = all_out_normaliser.normalise_array(all_out)

        LOG.info('Normalising done.')

    print '\n\n\nFirst 5 normalised'
    for i in range(0, 5):
        print all_in[i]
        print all_out[i]
        print galaxy_ids[i]
        print redshifts[i]

    all_in = np.array(all_in)
    all_out = np.array(all_out)

    mean_in = mean_values(all_in)
    mean_out = mean_values(all_out)
    std_in = std_dev(all_in)
    std_out = std_dev(all_out)

    print 'Mean std'
    for i in range(0, len(mean_in)):
        print'{0}   {1}'.format(mean_in[i], std_in[i])

    exit()

    test_in, train_in = split_data(all_in, config['test_data'])
    redshift_test, redshift_train = split_data(redshifts, config['test_data'])
    galaxy_ids_test, galaxy_ids_train = split_data(galaxy_ids, config['test_data'])
    test_out, train_out = split_data(all_out, config['test_data'])

    #for i in range(0, len(test_in)):
    #    print 'Galaxy ID: {0} Redshift: {1} test_in: {2} test_out: {3}\n'.format(galaxy_ids_test[i], redshift_test[i], test_in[i], test_out[i])

    """
    print 'Training in, out'
    print np.shape(train_in)
    print np.shape(train_out)
    print 'Testing in, out'
    print np.shape(test_in)
    print np.shape(test_out)
    """

    print '\n\n\nFirst test 5'
    for i in range(0, 5):
        print test_in[i]
        print test_out[i]
        print galaxy_ids_test[i]
        print redshift_test[i]

    # print 'Denormalised\n: '
    #for i in range(0, 43):
    #    print '{0}'.format(test_in_normaliser.denormalise_value(test_in[0][i], i)),

    return {'train_in': train_in, 'train_out': train_out, 'test_in': test_in, 'test_out': test_out,
            'galaxy_ids_test': galaxy_ids_test, 'galaxy_ids_train': galaxy_ids_train,
            'redshifts_test': redshift_test, 'redshifts_train': redshift_train,
            'in_normaliser': all_in_normaliser, 'out_normaliser': all_out_normaliser,
            'mean_in': mean_in, 'mean_out': mean_out,
            'stddev_in':std_in, 'stddev_out':std_out}


def write_dict(f, dictionary):
    for k, v in dictionary.iteritems():
        f.write('{0}            {1}\n'.format(k, v))
__author__ = 'ict310'
