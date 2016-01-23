import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import numpy as np
import abc
from common.logger import config_logger

LOG = config_logger(__name__)


class AbstractArrayNormaliser(object):
    """
    Base class for array normaliser
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def normalise(self, array):
        """
        Normalises the given array on the y axis
        Depends on implementation

        :param array: The array to normalise
        :return: Normalised array
        """
        return

    @abc.abstractmethod
    def denormalise(self, array):
        """
        Denormalises a whole array based on how the last array was normalised
        Depends on implementation

        :param array: Array to denormalise
        :return: denormalised array
        """
        return

    @abc.abstractmethod
    def denormalise_value(self, value, position):
        """
        Denormalises a single value that is part of one of the columns in the array
        Depends on implementation

        :param value: The value to denormalise
        :param position: The column in the array that the value is part of
        :return: Denormalised value
        """


class ArrayNormaliser(AbstractArrayNormaliser):
    def __init__(self):
        self.min = 0
        self.max = 1

        self.dataset_mins = None
        self.dataset_maxs = None

    def _scale_array_to(self, array, lower, upper):

        rng = self.dataset_maxs - self.dataset_mins

        return upper - (((upper - lower) * (self.dataset_maxs - array)) / rng)

    def normalise(self, array):

        self.dataset_maxs = np.max(array, axis=0)
        self.dataset_mins = np.min(array, axis=0)

        return self._scale_array_to(array, self.min, self.max)

    def denormalise(self, array):
        return self._scale_array_to(array, self.dataset_mins, self.dataset_maxs)

    def denormalise_value(self, value, position):
        """
        :param value the value to scale
        :param the y array index corresponds to the type of value that should be scaled (used to get min and max for that y set)
        """

        if value > self.max or value < self.min:
            return None

        return ((value - self.min) / (self.max - self.min)) * (self.dataset_maxs[position] - self.dataset_mins[position]) + self.dataset_mins[position]


class ArrayStandardiser(AbstractArrayNormaliser):
    """
    Implements standardisation for an array normaliser

    Ensures that the output means are 0 and standard deviation is 1
    """
    def __init__(self):
        self.means = None
        self.std = None

    def normalise(self, array):
        self.means = np.mean(array, axis=0)
        self.std = np.std(array, axis=0)
        return (array - self.means) / self.std

    def denormalise(self, array):
        return array * self.std + self.means

    def denormalise_value(self, value, position):
        return value * self.std[position] + self.means[position]


normalise = ArrayNormaliser
standardise = ArrayStandardiser


def get_normaliser(name):
    """
    Returns a normaliser class from the two standard normaliser classes

    :param name:
    :return:
    """
    class_type = getattr(sys.modules[__name__], name)

    return class_type()


def normaliser_from_user(user_input):
    """
    Determines if we can get a valid normaliser from the user's input
    :param user_input: the user's provided input. Can be a string or a class. Anything else is immediately invalid
    :return: True if a valid class can be derived from the user's input to be used as a normaliser
    """
    # If we got a string, check if it corresponds to a normaliser class
    if type(user_input) is str:
        try:
            return get_normaliser(user_input)
        except AttributeError as e:
            LOG.error('Invalid normaliser string provided: {0}'.format(user_input))
            raise e

    # If we got a class, check if it inherits from AbstractArrayNormaliser
    try:
        if isinstance(user_input, AbstractArrayNormaliser):
            return user_input
    except TypeError as e:
        LOG.error('Invalid type, cannot be interpreted as a normaliser: {0}'.format(type(user_input)))
        raise e

    # The previous code should act as a catch all, but just in case

    raise TypeError('Invalid type, cannot be interpreted as a normaliser: {0}'.format(type(user_input)))