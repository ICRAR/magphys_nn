# Magphys neural network preprocessor
# Date: 7/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Parser for the preprocessor
#
# The following pieces of data are extracted from a raw .fit file
#
# Input: fuv, nuv, u, g, r, y, z, Z, Y, J, H, K, WISEW1, WISEW2, WISEW3, WISEW4, PACS100, PACS160, SPIRE250, SPIRE350, SPIRE500
# The associated SNR values for each of these are also extracted.
# Output: i_sfh, i_ir, chi2, redshift
#         fmu(SFH), fmu(IR), mu, tauv, sSFR, M*, Ldust, T_W^BC, T_C^ISM, xi_C^tot, xi_PAH^tot, xi_MIR^tot, xi_W^tot, tvism, Mdus, SFR
#         fuv, nuv, u, g, r, y, z, Z, Y, J, H, K, WISEW1, WISEW2, WISEW3, WISEW4, PACS100, PACS160, SPIRE250, SPIRE350, SPIRE500
#

import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

from common import config
from common.logger import config_logger
import os

LOG = config_logger('__name__')

process_data_input_key = [
    'galaxy_name',
    'redshift',
    'fuv',
    'nuv',
    'u',
    'g',
    'r',
    'z',
    'Z',
    'Y',
    'J',
    'H',
    'K',
    'WISEW1',
    'WISEW2',
    'WISEW3',
    'WISEW4',
    'PACS100',
    'PACS160',
    'SPIRE250',
    'SPIRE350',
    'SPIRE500',
    'fuv_snr',
    'nuv_snr',
    'u_snr',
    'g_snr',
    'r_snr',
    'z_snr',
    'Z_snr',
    'Y_snr',
    'J_snr',
    'H_snr',
    'K_snr',
    'WISEW1_snr',
    'WISEW2_snr',
    'WISEW3_snr',
    'WISEW4_snr',
    'PACS100_snr',
    'PACS160_snr',
    'SPIRE250_snr',
    'SPIRE350_snr',
    'SPIRE500_snr'
]

class NaNValue(Exception):
    """
    Exception fired when a NaN value is found in a fit file
    and the config option to accept NaN values is set to False
    """
    def __init__(self, value):
        self.value = value
        self.message = value

    def __str__(self):
        return repr(self.value)


class InvalidFile(Exception):
    """
    Exception fired when a file is deemed invalid
    e.g. not containing the .fit header we're looking for
    """
    def __init__(self, value):
        self.value = value
        self.message = value

    def __str__(self):
        return repr(self.value)


def string2float(string_list):
    """
    Converts a string list to a float list

    :param string_list: the string list to convert
    :return:
    """
    try:
        out = []
        for item in string_list:
            out.append(float(item))
    except ValueError:
        LOG.exception('Error converting string to float')
        return None

    return out


def import_to_dict(keys, values):
    """
    Converts two lists to a dict

    :param keys: keys in the dict
    :param values: values in the dict
    :return: Dictionary mapping keys to values
    """

    if len(keys) is not len(values):
        LOG.error('Cannot map {0} keys to {1} values'.format(len(keys), len(values)))
        return None

    # Correct mapping of keys to values
    new_dict = {}

    for i in range(0, len(keys)):
        new_dict[keys[i]] = values[i]

    return new_dict


def check_values(values):
    """
    Checks all the values in a list and determines if they are to be considered valid
    :param values:
    :return:
    """

    if config.ALLOW_NAN:
        return True

    for value in values:
        if float(value) == 0:# or np.isnan(value):
            return False

    return True


def check_directory(directory):
    """
    Checks a directory and returns true if it contains at least one .fit file
    :param directory:
    :return:
    """
    files = os.listdir(directory)

    for single_file in files:
        if single_file.endswith('.fit'):
            return True

    return False


def parse_process_file(filename):
    """
    Parses the process_data.sh file that comes along side a .fit file.
    Contains the redshift of the galaxy along with all flux values in Janskys (Jy)
    :param filename:
    :return:
    """

    # First things first! These values are useless unless theres at least one .fit file in the same directory
    path, filename_t = os.path.split(filename)

    if not check_directory(path):
        raise InvalidFile('No .fit files found in directory {0}. Cannot parse process_data.sh'.format(path))

    galaxy_next = False
    valid_file = False
    galaxies = []

    with open(filename, 'r') as process:
        for line in process:

            if galaxy_next:
                stripped = line.strip()

                if stripped == '' or not stripped.startswith('echo'):
                    galaxy_next = False
                    continue

                # Remove this crap
                stripped.strip('echo "').strip('" >> mygals.dat')

                values = stripped.split()

                galaxies.append(dict(zip(process_data_input_key, values)))

            if line.startswith('echo "# Header" > mygals.dat'):
                galaxy_next = True
                valid_file = True
                continue

    if valid_file:
        return galaxies
    else:
        raise InvalidFile('Not a valid process_data.sh file, or empty')


def parse_fit_file(filename):

    output_dict_percentiles = {}

    skip_lines = 0

    valid_file = False

    # These flags all used to parse info from the file in the correct order.
    # Faster than doing constant string checks for everything
    input_keys_next = False
    input_values_next = False
    input_values_snr_next = False

    output_best_fit_model_next = False

    output_best_fit_keys_next = False
    output_best_fit_values_next = False

    output_best_fit_inputs_next = False  # Output section in the file that has the exact same parameter names as the inputs

    percentiles = False
    percentile_values_next = False
    # -------------------------End flags-------------------------

    with open(filename, 'r') as fit:
        for line in fit:

            if skip_lines > 0:
                skip_lines -= 1
                continue

            # -----------------------------Block for capturing inputs and best fits-----------------------------
            if not percentiles:

                if line.startswith('# OBSERVED FLUXES (and errors):'):  # 1 Finding first line
                    input_keys_next = True
                    valid_file = True
                    continue

                if input_keys_next:  # 2 Column headings for input
                    line_inputs = line.strip('# \n').split()
                    input_keys = list(line_inputs)

                    input_keys_next = False
                    input_values_next = True
                    continue

                if input_values_next:  # 3 Input values
                    line_inputs = line.strip(' \n').split()
                    input_values = list(line_inputs)
                    if not check_values(input_values):
                        raise NaNValue('Contains a NaN!')

                    input_values_next = False
                    input_values_snr_next = True
                    # We now have input keys and input values ready for storing in a dict
                    continue

                if input_values_snr_next:  # 4 Input signal to noise values
                    line_inputs = line.strip(' \n').split()
                    input_values_snr = list(line_inputs)
                    if not check_values(input_values_snr):
                        raise NaNValue('Contains a NaN!')

                    input_values_snr_next = False
                    # We now have input keys and input values ready for storing in a dict
                    continue

                if line.startswith('# BEST FIT MODEL: (i_sfh, i_ir, chi2, redshift)'):  # 5 Header for the part containing best fit values
                    output_best_fit_model_next = True
                    continue

                if output_best_fit_model_next:  # 6 Pull the values under # BEST FIT MODEL: (i_sfh, i_ir, chi2, redshift)
                    line_inputs = line.strip(' \n').split()
                    output_best_fit_model_values = list(line_inputs)

                    output_best_fit_model_next = False
                    output_best_fit_keys_next = True
                    continue

                if output_best_fit_keys_next:  # 7 Grab the keys for the best fit
                    line_inputs = line.strip('# \n').split('.')

                    t = [y for y in line_inputs if y != '']
                    del line_inputs[:]
                    line_inputs.extend(t)
                    output_best_fit_keys = list(line_inputs)

                    output_best_fit_keys_next = False
                    output_best_fit_values_next = True
                    continue

                if output_best_fit_values_next:  # 8 Get the values for the best fit
                    line_inputs = line.strip('').split()
                    output_best_fit_values = list(line_inputs)

                    # We now have best fit outputs and values ready for storing in a dict

                    output_best_fit_values_next = False
                    output_best_fit_inputs_next = True
                    skip_lines = 1
                    continue

                if output_best_fit_inputs_next:  # 9 Get the output values that directly match the input values (do we even need these?)
                    line_inputs = line.strip('').split()
                    output_best_fit_inputs = list(line_inputs)

                    output_best_fit_inputs_next = False
                    percentiles = True  # We're now looking for percentiles
                    continue
            # -----------------------------Percentiles-----------------------------
            else:
                if line.startswith('# ...'):
                    line_inputs = line.split('...')
                    current_param_name = line_inputs[1].strip()
                elif line.startswith('#....percentiles of the PDF......'):
                    percentile_values_next = True
                elif percentile_values_next:
                    line_inputs = line.split()
                    output_dict_percentiles[current_param_name] = list(line_inputs)
                    percentile_values_next = False

    # LOG.info('Parsing complete for {0}'.format(filename))

    # Don't worry about the 'might be referenced before assignment' bits here. If valid_file = true, then these files have been found
    if valid_file:
        inputs_dict = import_to_dict(input_keys, string2float(input_values))
        inputs_snr_dict = import_to_dict(input_keys, string2float(input_values_snr))

        outputs_best_fit_model_dict = import_to_dict(['i_sfh', 'i_ir', 'chi2', 'redshift'], string2float(output_best_fit_model_values))
        outputs_best_fit_dict = import_to_dict(output_best_fit_keys, string2float(output_best_fit_values))
        outputs_best_fit_inputs = import_to_dict(input_keys, string2float(output_best_fit_inputs))

        outputs_median_values = {}

        # Package the median outputs only into a dict.
        # Can change the '2' here to something else if we want another percentile
        for k, v in output_dict_percentiles.iteritems():
            outputs_median_values[k] = string2float(v)[2]

        return inputs_dict, inputs_snr_dict, outputs_median_values, outputs_best_fit_model_dict, outputs_best_fit_dict, outputs_best_fit_inputs
    else:
        raise InvalidFile('No data')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='file', nargs=1, help='Path to the fit file')

    args = vars(parser.parse_args())
    fit_file = args['file'][0]

    parse_fit_file(fit_file)






