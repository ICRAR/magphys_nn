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
from binning import get_binning_function

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


def recursive_write_dict(f, dictionary, prepend=''):
    """
    Writes a dictionary out to the specified file.
    If the dictionary contains nested dictionaries, they are printed out on a different row
    :param f:
    :param dictionary:
    :param prepend:
    :return:
    """
    for k, v in dictionary.iteritems():
        if type(v) is dict:
            f.write('{0}\n'.format(k))
            recursive_write_dict(f, v, '{0}\t'.format(prepend))
        else:
            f.write('{0}{1}\t\t{2}\n'.format(prepend, k, v))


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


def remove_above_percentile(all_in, all_out, redshifts, galaxy_ids, percentile):
    percentiles = np.percentile(all_in, percentile, axis=0)

    bad_ids = set() # set is useful for no repeats
    for i in range(0, len(all_in[0])):
        tmp_list = np.where(all_in[:,i] > percentiles[i])[0]
        bad_ids.update(tmp_list)

    print 'Bad IDs based on percentile {0}: {1}'.format(percentile, len(bad_ids))

    bad_ids = list(bad_ids) # numpy doesn't like sets
    all_in = np.delete(all_in, bad_ids, axis=0)
    all_out = np.delete(all_out, bad_ids, axis=0)
    redshifts = np.delete(redshifts, bad_ids, axis=0)
    galaxy_ids = np.delete(galaxy_ids, bad_ids, axis=0)

    return all_in, all_out, redshifts, galaxy_ids


def remove_negative_values(all_in, all_out, redshifts, galaxy_ids):
    # Remove negative fluxes
    bad_ids = set()
    for i in range(0, len(all_in[0])):
        tmp_list = np.where(all_in[:,i] < 0)[0]
        bad_ids.update(tmp_list)

    LOG.info('****Removing {0} total negative input values****'.format(len(bad_ids)))

    bad_ids = list(bad_ids)  # numpy doesn't like sets
    all_in = np.delete(all_in, bad_ids, axis=0)
    all_out = np.delete(all_out, bad_ids, axis=0)
    redshifts = np.delete(redshifts, bad_ids, axis=0)
    galaxy_ids = np.delete(galaxy_ids, bad_ids, axis=0)

    return all_in, all_out, redshifts, galaxy_ids


def shuffle_arrays(*args):
    """
    Shuffles all of the given arrays the same way.

    Arrays must be the same length and must be numpy arrays
    """
    if not args:
        # passed nothing
        return

    # Make a random permutation
    perm = np.random.permutation(len(args[0]))

    out_tuple = list()
    for count, item in enumerate(args):
        # Apply permutation to each item, then append them to a list for outputting
        out_tuple.append(item[perm])

    return tuple(out_tuple)


def random_in_shape(array, low=0, high=1):
    """
    Return a set of random numbers in the same shape as array
    :param array:
    :param low:
    :param high:
    :return:
    """

    shape = np.shape(array)
    return np.random.uniform(low, high, shape)


def get_training_data(DatabaseConfig, PreprocessingConfig, FileConfig):

    tmp_file = FileConfig['temp_file']
    erase_above = PreprocessingConfig['erase_above']
    remove_negatives = PreprocessingConfig['remove_negative_inputs']
    flip = PreprocessingConfig['flip']
    random_input = PreprocessingConfig['random_input']
    test_data = DatabaseConfig['test_data']

    bin_type = PreprocessingConfig['binning_type']
    bin_precision = PreprocessingConfig['binning_precision']

    single_input = PreprocessingConfig['single_input']
    single_output = PreprocessingConfig['single_output']

    input_normaliser = PreprocessingConfig['normalise_input']
    output_normaliser = PreprocessingConfig['normalise_output']

    if check_temp(tmp_file, DatabaseConfig): # Check the temp file

        LOG.info('Correct temp file exists at {0}, loading from temp'.format(tmp_file))
        all_in, all_out, redshifts, galaxy_ids = load_from_file(tmp_file)
        LOG.info('Done.')
    else:

        LOG.info('No temp file, reading from database.')
        all_in, all_out, redshifts, galaxy_ids = get_train_test_data(DatabaseConfig)

        LOG.info('Done. Writing temp file for next time.')
        write_file(tmp_file, DatabaseConfig, all_in, all_out, redshifts, galaxy_ids)
        LOG.info('Done. Temp file written to {0}'.format(tmp_file))

    # Convert the arrays to np arrays
    all_in = np.array(all_in)
    all_out = np.array(all_out)
    redshifts = np.array(redshifts)
    galaxy_ids = np.array(galaxy_ids)

    # Flip input and output
    if flip:
        LOG.info('****Flipping input and output!****')
        tmp = all_in
        all_in = all_out
        all_out = tmp

    # Use random values for input
    if random_input:
        LOG.info('****Using randomly generated input values!****')
        all_in = random_in_shape(all_in)

    # Do we want only a single output/input?
    if single_output is not None:
        LOG.info('****Using single output {0}****'.format(single_output))
        all_out = all_out[:,single_output]

    if single_input is not None:
        LOG.info('****Using single input {0}****'.format(single_input))
        all_in = all_in[:,single_input]

    # Remove negative values in the input (BEFORE normalisation!)
    if remove_negatives:
        LOG.info('****Removing negative input values****'.format(single_input))
        all_in, all_out, redshifts, galaxy_ids = remove_negative_values(all_in, all_out, redshifts, galaxy_ids)

    # Remove negative flux and values above percentile 99
    if erase_above is not None:
        LOG.info('****Removing all input values above percentile {0}'.format(erase_above))
        all_in, all_out, redshifts, galaxy_ids = remove_above_percentile(all_in, all_out, redshifts, galaxy_ids
                                                                         , erase_above)
    # Bin the values in to percentile groups
    if bin_type is not None:
        LOG.info('****Binning all input and output values at precision {0}****'.format(bin_precision))
        bin_func = get_binning_function(bin_type)
        all_in = bin_func(all_in, bin_precision)
        all_in = all_in.astype('float32')

        all_out = bin_func(all_out, bin_precision)
        all_out = all_out.astype('float32')
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
        print galaxy_ids[i]
        print redshifts[i]
        print all_in[i]
        print all_out[i]
        print
    """

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
    print '\n\n\nFirst 5'
    for i in range(0, 5):
        print galaxy_ids[i]
        print redshifts[i]
        print all_in[i]
        print all_out[i]
        print
    """

    # Grab some statistics of everything.
    mean_in = np.mean(all_in, axis=0)
    mean_out = np.mean(all_out, axis=0)
    std_in = np.std(all_in, axis=0)
    std_out = np.std(all_out, axis=0)

    min_in = np.min(all_in, axis=0)
    max_in = np.max(all_in, axis=0)

    min_out = np.min(all_out, axis=0)
    max_out = np.max(all_out, axis=0)

    # Shuffle all arrays in the same way
    all_in, all_out, redshifts, galaxy_ids = shuffle_arrays(all_in, all_out, redshifts, galaxy_ids)

    # Split the data up in to training and test sets.
    split_point = test_data
    test_in, train_in = split_data(all_in, split_point)
    redshift_test, redshift_train = split_data(redshifts, split_point)
    galaxy_ids_test, galaxy_ids_train = split_data(galaxy_ids, split_point)
    test_out, train_out = split_data(all_out, split_point)

    # A lot of data to return
    return {'train_in': train_in, 'train_out': train_out, 'test_in': test_in, 'test_out': test_out,
            'galaxy_ids_test': galaxy_ids_test, 'galaxy_ids_train': galaxy_ids_train,
            'redshifts_test': redshift_test, 'redshifts_train': redshift_train,
            'in_normaliser': in_normaliser, 'out_normaliser': out_normaliser,
            'mean_in': mean_in, 'mean_out': mean_out,
            'stddev_in':std_in, 'stddev_out':std_out,
            'min_in':min_in, 'max_in':max_in,
            'min_out':min_out, 'max_out':max_out}
