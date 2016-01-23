# Shared neural network functions

import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import numpy as np
from common.database import get_train_test_data

import pickle
from keras.callbacks import Callback
from common.logger import config_logger
from normalisation import normaliser_from_user

LOG = config_logger(__name__)


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


def write_dict(f, dictionary):
    for k, v in dictionary.iteritems():
        f.write('{0}            {1}\n'.format(k, v))


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


def get_training_data(config, tmp_file, single_output=None, single_input=None,
                      input_normaliser=None, output_normaliser=None,
                      unknown_input_handler=None):
    """
    Gets sets of training and test data for use in a neural network.
    All returned as numpy arrays.
    Return dictionary has the following elements:

    train_in: Training data input data. Shape (total_sets - testing_sets, number_input_features)
    train_out: Training output data. Shape (total_sets - testing, number_output_features
    test_in:
    test_out
    galaxy_ids_test:
    galaxy_ids_train:
    redshifts_train:
    redshifts_test:
    in_normaliser:
    out_normaliser:
    mean_in:
    mean_out:
    stddev_in:
    stddev_out:

    :param config:
    :param tmp_file:
    :param single_output:
    :param single_input:
    :param input_normaliser:
    :param output_normaliser:
    :param unknown_input_handler:
    :return:
    """

    # This dictionary is what we use to check whether we need to re-query the database.
    # If this exact same dict is read from the temp file, then the temp file is fine.
    # If it is not, we need to re-query the database
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

    # Convert the arrays to np arrays
    all_in = np.array(all_in)
    all_out = np.array(all_out)
    redshifts = np.array(redshifts)
    galaxy_ids = np.array(galaxy_ids)

    """
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
    """

    # Do we want only a single output/input?
    if single_output is not None and single_output > 0:
        all_out = all_out[:,single_output]

    if single_input is not None and single_input > 0:
        all_in = all_in[:,single_input]

    in_normaliser = None
    out_normaliser = None

    if input_normaliser is not None:
        # We should normalise the input
        LOG.info('Normalising input with {0}'.format(input_normaliser))

        # Get the normaliser class specified by the user
        in_normaliser = normaliser_from_user(input_normaliser)

        all_in = in_normaliser.normalise(all_in)

        LOG.info('Normalising input done.')

    if output_normaliser is not None:
        # We should normalise the output
        LOG.info('Normalising output with {0}'.format(output_normaliser))

        # Get the normaliser class specified by the user
        out_normaliser = normaliser_from_user(output_normaliser)

        all_out = out_normaliser.normalise(all_out)

        LOG.info('Normalising output done.')

    """
    print '\n\n\nFirst 5 normalised'
    for i in range(0, 5):
        print all_in[i]
        print all_out[i]
        print galaxy_ids[i]
        print redshifts[i]
    """

    mean_in = np.mean(all_in, axis=0)
    mean_out = np.mean(all_out, axis=0)
    std_in = np.std(all_in, axis=0)
    std_out = np.std(all_out, axis=0)

    print '\n\n\nFirst 5 normalised'
    for i in range(0, 5):
        print all_in[i]
        print all_out[i]
        print galaxy_ids[i]
        print redshifts[i]

    print 'Mean in'
    print mean_in
    print 'Mean out'
    print mean_out
    print 'Std dev in'
    print std_in
    print 'Std dev out'
    print std_out

    print '\n\n\nFirst 5 de-normalised'
    for i in range(0, 5):
        print in_normaliser.denormalise(all_in[i])
        print out_normaliser.denormalise(all_out[i])
        print galaxy_ids[i]
        print redshifts[i]

    exit()

    # Split the data up in to training and test sets.
    split_point = config['test_data']
    test_in, train_in = split_data(all_in, split_point)
    redshift_test, redshift_train = split_data(redshifts, split_point)
    galaxy_ids_test, galaxy_ids_train = split_data(galaxy_ids, split_point)
    test_out, train_out = split_data(all_out, split_point)

    """
    print '\n\n\nFirst test 5'
    for i in range(0, 5):
        print test_in[i]
        print test_out[i]
        print galaxy_ids_test[i]
        print redshift_test[i]
    """

    return {'train_in': train_in, 'train_out': train_out, 'test_in': test_in, 'test_out': test_out,
            'galaxy_ids_test': galaxy_ids_test, 'galaxy_ids_train': galaxy_ids_train,
            'redshifts_test': redshift_test, 'redshifts_train': redshift_train,
            'in_normaliser': in_normaliser, 'out_normaliser': out_normaliser,
            'mean_in': mean_in, 'mean_out': mean_out,
            'stddev_in':std_in, 'stddev_out':std_out}
