import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

from common.logger import config_logger
import numpy as np

LOG = config_logger(__name__)

# Set of unknown input handlers


def replace_zeros(array, ignore=None):
    """
    Replaces all bad inputs with 0
    :param array:
    :param ignore:
    :return:
    """
    for i in range(0, len(array[0])):
        if i == ignore:
            continue
        column = array[:,i]
        column[column < 0] = 0

    return array


def replace_mean(array, ignore=None):
    """
    Replaces all bad inputs with the mean value.
    :param array:
    :param ignore:
    :return:
    """
    means = np.mean(array, axis=0)

    for i in range(0, len(array[0])):
        if i == ignore:
            continue
        column = array[:,i]
        column[column < 0] = means[i]

    return array


def get_unknown_hanlder(name):
    """
    Returns an unknown input handler

    :param name:
    :return:
    """
    return getattr(sys.modules[__name__], name)