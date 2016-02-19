import ConfigParser
import os

# Config section names to be written/read from config file. Must match the dictionaries following.
sections = ['DatabaseConfig', 'PreprocessingConfig', 'LearningParameters', 'TrainingParameters', 'NetworkStructure', 'FileConfig']

# Config defaults
DatabaseConfig = {'database_connection_string': 'sqlite:///Database_run06.db',
                  'train_data': 200000,
                  'test_data': 1000,
                  'run_id': '06',
                  'output_type': 'median',  # median, best_fit, best_fit_model, best_fit_inputs
                  'input_type': 'normal',  # normal, Jy
                  'include_sigma': False,  # True, False
                  'unknown_input_handler': None,
                  'input_filter_types': None
                  }

PreprocessingConfig = {'single_input': None,
                       'single_output': None,
                       'normalise_input': 'standardise',
                       'normalise_output': None,
                       'binning_precision': 0.01,  # bin values in to percentile groups (0 - 10, 10 - 20 etc.)
                       'binning_type': None, # percentile, absolute
                       'erase_above': None, # Erase all values above this percentile. (remove outliers)
                       'random_input': False, # Use random values as inputs instead of actual inputs.
                       'flip': False, # Flip the inputs and outputs, so outputs become inputs and inputs become outputs
                       'remove_negative_inputs': False # remove all negative inputs.
                      }

LearningParameters = {'momentum': 0.0,
                      'learning_rate': 0.01,
                      'decay': 0
                      }

TrainingParameters = {'batch_size': 1000,
                      'epochs_per_fit': 10,
                      'validation_split': 0.3,
                      'max_epochs': 1000,
                      'num_tests': 30
                      }

NetworkStructure = {'input_activation': 'relu',  # softmax, softplus, relu, tanh, sigmoid, hard_sigmoid, linear
                    'hidden_activation': 'relu',
                    'output_activation': 'linear',
                    'use_dropout': True, 'dropout_rate': 0.5,  # 0 to 1
                    'use_batch_norm': False,
                    'initialisation_type': 'glorot_normal',  # uniform, lecun_uniform, normal, identity, orthogonal, zero, glorot_normal, glorot_uniform, he_normal, he_uniform
                    'hidden_connections': 300,
                    'hidden_layers':2,
                    'loss': 'mse',
                    'optimiser': 'sgd'
                    }

FileConfig = {'temp_file': 'nn_last_tmp_input.tmp',
              'output_directory': '',
              'output_file_name': 'output'
              }


def dict2section(c, sec_name, d):
    """
    Converts a dictionary to a config section
    :param c: Config object
    :param sec_name: Name of the section
    :param d: Dictionary to convert to section
    :return:
    """
    c.add_section(sec_name)

    for k, v in d.iteritems():
        c.set(sec_name, k, v)


def recursive_set_list_types(l):

    for i, item in enumerate(l):
        try:
            flt = float(item)
            integer = int(item)

            # If a float representation and an int representation are equal
            # it's acceptable to represent this as an int.
            # Otherwise it must be a float.
            if flt != integer:
                l[i] = flt
            else:
                l[i] = integer
            continue
        except:
            pass

        if item == 'None':
            l[i] = None

        if item.lower() == 'True'.lower():
            l[i] = True
            continue

        if item.lower() == 'False'.lower():
            l[i] = False
            continue

        if item.startswith('['):
            if item.endswith(']'):
                l[i] = recursive_set_list_types(item[1:-1].split(','))
            else:
                raise Exception('Incorrectly terminated list: {0}'.format(item))
    return l


def set_dict_types(d):
    # Painful type setting
    for k, v in d.iteritems():

        try:
            d[k] = int(v)
            continue
        except:
            pass

        try:
            d[k] = float(v)
            continue
        except:
            pass

        if v == 'None':
            d[k] = None
            continue

        if v.lower() == 'True'.lower():
            d[k] = True
            continue

        if v.lower() == 'False'.lower():
            d[k] = False
            continue

        if v.startswith('['):
            if v.endswith(']'):
                d[k] = recursive_set_list_types(v[1:-1].split(','))
                continue
            else:
                raise Exception('Incorrectly terminated list: {0}'.format(v))
    return d


def sect2dict(c, sec_name):
    """
    Convert a config section to a dictionary
    :param c: Config object
    :param sec_name: Name of the section to read
    :return:
    """
    from_config = set_dict_types(dict(c.items(sec_name)))
    # We should also have a default dict for this section.

    # Compare the read in section to the default and ensure
    # all of the keys in the default are present in the read in dict

    for key in globals()[sec_name]:
        if key not in from_config:
            raise Exception('Missing required key "{0}" in section "{1}"'.format(key, sec_name))
    return from_config


def read_config(path):
    """
    Read a config file a the specified path
    :param path: Path to the config file
    :return:
    """
    config = ConfigParser.RawConfigParser()
    config_dict = {}

    with open(path, 'r') as config_file:
        config.readfp(config_file)

        for section in sections:  # This list contains all the section names we require from the config
            if config.has_section(section):  # Check if the required section is there
                config_dict[section] = sect2dict(config, section)  # Load it if it is
            else:
                raise Exception('Could not find required section {0} in config file at {1}'.format(section, path))

    return config_dict


def make_default(name, directory=None):
    """
    Makes a default config file
    :param directory: Directory to make the config file in
    :return:
    """
    config = ConfigParser.RawConfigParser()

    # Section names
    for section in sections:
        dict2section(config, section, globals()[section])  # :3

    if directory:
        name = '{0}/{1}'.format(directory.strip('\/'), name)

    with open(name, 'wb') as config_file:
        config.write(config_file)
    os.chmod(name, 0o777)


if __name__ == '__main__':
    import sys, os

    if len(sys.argv) == 1:
        make_default('default.cfg')
    else:

        if os.path.isdir(sys.argv[1]):
            make_default('default.cfg',sys.argv[1])
        else:
            raise Exception('Invalid directory specified')