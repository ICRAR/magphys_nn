# Magphys neural network preprocessor
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Logging functions for the preprocessor
# Modified from https://github.com/ICRAR/boinc-magphys/blob/master/server/src/utils/logging_helper.py
#

"""
Configure a logger
"""
import logging
import logging.handlers

# Set up the root logger as the project likes it
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s:' + logging.BASIC_FORMAT)


def config_logger(name):
    """
    Get a logger
    :param name:
    :return:
    """
    logger = logging.getLogger(name)

    return logger


def add_file_handler_to_root(file_name):
    """
    Added a file logger to the root
    :param file_name:
    :return:
    """
    formatter = logging.Formatter('%(asctime)-15s:' + logging.BASIC_FORMAT)
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
