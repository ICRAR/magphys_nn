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
from common import config
from common.logger import config_logger

LOG = config_logger('__name__')


class NaNValue(Exception):
    def __init__(self, value):
        self.value = value
        self.message = value

    def __str__(self):
        return repr(self.value)


class InvalidFile(Exception):
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


def parse_file(filename):
    # Parse through a .fit file and pull out what we want.

    # Search for next line:
    # OBSERVED FLUXES (and errors):
    #   fuv         nuv         u           g           r           i           z           Z           Y           J           H           K           WISEW1	 WISEW2      WISEW3	 WISEW4      PACS100     PACS160     SPIRE250    SPIRE350    SPIRE500
    # LINE WITH INPUT
    # LINE WITH INPUT SNR (If either of these have NaNs in it, check the config as to whether we can skin NaNs

    # Search for next line:
    # BEST FIT MODEL: (i_sfh, i_ir, chi2, redshift)
    # FOUR VALUES FOR RESULT
    # #.fmu(SFH)...fmu(IR)........mu......tauv........sSFR..........M*.......Ldust......T_W^BC.....T_C^ISM....xi_C^tot..xi_PAH^tot..xi_MIR^tot....xi_W^tot.....tvism.......Mdust.....SFR
    # LINE WITH OUTPUT
    #   fuv         nuv         u           g           r           i           z           Z           Y           J           H           K           WISEW1	 WISEW2      WISEW3	 WISEW4      PACS100     PACS160     SPIRE250    SPIRE350    SPIRE500
    # LINE WITH OUTPUT

    # Next search through for the line:
    #....percentiles of the PDF......
    # OUTPUT VALUES
    # Pull the output values out for each of these instances.

    output_dict_percentiles = {}

    skip_lines = 0

    valid_file = False

    input_keys_next = False
    input_values_next = False
    input_values_snr_next = False

    output_best_fit_model_next = False

    output_best_fit_keys_next = False
    output_best_fit_values_next = False

    output_best_fit_inputs_next = False  # Output section in the file that has the exact same parameter names as the inputs

    percentiles = False
    percentile_values_next = False

    with open(filename, 'r') as fit:
        for line in fit:

            if skip_lines > 0:
                skip_lines -= 1
                continue

            if not percentiles:  # Block for capturing inputs and best fits

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

    parse_file(fit_file)






