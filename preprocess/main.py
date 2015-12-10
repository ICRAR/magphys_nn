# Magphys neural network preprocessor
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Preprocessor program to extract desired input and outputs from raw .fit files.
# Extracted inputs and outputs are placed into an SQL database
#
# The following pieces of data are extracted from a raw .fit file
#
# Input: fuv, nuv, u, g, r, y, z, Z, Y, J, H, K, WISEW1, WISEW2, WISEW3, WISEW4, PACS100, PACS160, SPIRE250, SPIRE350, SPIRE500
# The associated SNR values for each of these are also extracted.
# Output: i_sfh, i_ir, chi2, redshift
#         fmu(SFH), fmu(IR), mu, tauv, sSFR, M*, Ldust, T_W^BC, T_C^ISM, xi_C^tot, xi_PAH^tot, xi_MIR^tot, xi_W^tot, tvism, Mdus, SFR
#         fuv, nuv, u, g, r, y, z, Z, Y, J, H, K, WISEW1, WISEW2, WISEW3, WISEW4, PACS100, PACS160, SPIRE250, SPIRE350, SPIRE500
#
# This program will pull out both the best fit for the results as well as the median.
#

import os
import sys
import config
import argparse
import random as rand
from logger import config_logger
from parser import parse_file
from database import add_to_db, exists_in_db
import datetime

LOG = config_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-dir', dest='working_directory', nargs=1, help='Path to the base directory containing the .fit files. (/mnt/gamad/kevin/)')
parser.add_argument('-n', dest='num_to_load', type=int, nargs=1, help='Number of galaxies to load')
parser.add_argument('-r', dest='run_id', nargs='*', help='Run ID folders to load. If ommitted, simply searches for all .fit files in subdirectories of the working directory.')

args = vars(parser.parse_args())
working_directory = args['working_directory'][0]
num_to_load = args['num_to_load'][0]
run_ids = args['run_id']

run_dirs = []
num_added = 0
current_run_dir = ''

# Is everything valid?
if os.path.exists(working_directory):
    # Check each of the run_ids and check if they're valid
    invalid_folders = []
    for id in run_ids:
        full_path = os.path.join(working_directory, id)
        run_dirs.append(full_path)
        if not os.path.exists(full_path):
            invalid_folders.append(full_path)

    if len(invalid_folders) > 0:
        LOG.error('The folllowing specified run folders do not exist: ')

        for invalid in invalid_folders:
            LOG.error('{0}'.format(invalid))

        exit(0)


def action_null(filename):
    LOG.info(filename)
    return False


def on_file(filename):
    if filename.endswith('.fit'):

        if not exists_in_db(filename) or config.RELOAD:
            try:
                inputs, inputs_snr, median_outputs = parse_file(filename)

                details = {}
                details['run_id'] = current_run_dir
                details['filename'] = filename
                details['last_updated'] = datetime.datetime.now()
                details['type'] = 'median'

                add_to_db(inputs, inputs_snr, median_outputs, details)

                global num_added  # ???
                num_added += 1

                if num_added is num_to_load:
                    return True

            except Exception as e:
                LOG.error('File {0} {1}'.format(filename, e.message))

    return False


def recursive_dir_walk(root, file_action=action_null, shuffle=False):
    for dirname, dirnames, filenames in os.walk(root):
        if shuffle:
            rand.shuffle(dirnames)
            rand.shuffle(filenames)

        for single_file in filenames:
            LOG.info('File {0}'.format(single_file))
            if file_action(os.path.join(dirname, single_file)):
                return 1

        for directory in dirnames:
            # Recurse in to the other directories
            LOG.info('Entering {0}'.format(directory))
            if recursive_dir_walk(directory, file_action, shuffle):
                # Returned true, we've got what we need
                return 1

        return 0  # Gone through all files

# Get the required files from each of the run directories
for directory in run_dirs:

    # on_file is called when we hit a file. In that function we work out what to do with the file.
    current_run_dir = directory
    if recursive_dir_walk(directory, on_file, config.SHUFFLE):
        LOG.info('Found all required files {0}. Runs added to database.'.format(num_added))
    else:
        LOG.info('Could not add the desired number of files. Added {0} instead'.format(num_added))




















