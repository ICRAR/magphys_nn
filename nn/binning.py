import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

from common.logger import config_logger
import numpy as np

LOG = config_logger(__name__)


def get_percentile_groups(data, percentile_step):

    binned = None
    percentiles_list = np.arange(0, 100, percentile_step*100)
    np.append(percentiles_list, 100)

    single = False

    try:
        length = len(data[0])
    except:
        # Single array
        length = 1
        single = True

    for i in range(0, length):
        binned_column = []

        if single:
            percentiles = np.percentile(data, percentiles_list)
            n_data = data
        else:
            percentiles = np.percentile(data[:,i], percentiles_list)
            n_data = data[:,i]

        for value in n_data:
            # For each value in the row, assign a binned value

            for lower in range(0, len(percentiles)):
                upper = lower + 1

                if upper == len(percentiles) - 1: # Made it all the way through all the categories
                    binned_column.append(lower)
                    break

                if percentiles[lower] <= value < percentiles[upper]:
                    binned_column.append(lower) # Lower is bin number
                    break

        if binned is None:
            binned = binned_column
        else:
            binned = np.c_[binned, binned_column]

    return np.array(binned)


def get_dataspace_cutoffs(data, size):
    cutoffs = []

    dataspace_min = min(data)
    dataspace_max = max(data)

    difference = dataspace_max - dataspace_min

    i = 0
    result = 0

    cutoffs.append(dataspace_min)
    while result < dataspace_max:
        i += 1
        result = size* i * difference + dataspace_min
        cutoffs.append(result)

    return cutoffs


def get_dataspace_groups(data, step):

    binned = None
    single = False

    try:
        length = len(data[0])
    except:
        # Single array
        length = 1
        single = True

    for i in range(0, length):
        binned_column = []

        if single:
            cutoffs = get_dataspace_cutoffs(data, step)
            n_data = data
        else:
            cutoffs = get_dataspace_cutoffs(data[:,i], step)
            n_data = data[:,i]

        for value in n_data:
            # For each value in the row, assign a binned value

            for lower in range(0, len(cutoffs)):
                upper = lower + 1

                if upper == len(cutoffs):  # Made it all the way through all the categories
                    binned_column.append(lower)
                    break

                if cutoffs[lower] <= value < cutoffs[upper]:
                    binned_column.append(lower)  # Lower is bin number
                    break

        if binned is None:
            binned = binned_column
        else:
            binned = np.c_[binned, binned_column]

    return np.array(binned)


# only for single outputs
def bin2class(binned):
    classes = []

    minimum = min(binned)
    maximum = max(binned)

    rng = maximum - minimum

    for item in binned:
        zeros = np.zeros(rng)
        np.put(zeros, item-1, 1)
        classes.append(zeros)

    return np.array(classes)


percentile = get_percentile_groups
absolute = get_dataspace_groups


def get_binning_function(name):
    """
    Returns an unknown input handler

    :param name:
    :return:
    """
    return getattr(sys.modules[__name__], name)