# Magphys neural network
# Date: 10/12/2015
# Author: Samuel Foster
# Organisation: ICRAR
#
# Neural network to perform SED fitting
#
# Input: SED inputs (fuv, nuv, u, g etc.)
# Output: Median fit values
import os, sys

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import random as rand
import numpy as np
from common.database import get_train_test_data, db_init
import pickle
from pybrain.structure import *
from pybrain.supervised import *
from pybrain.datasets import *
from common.logger import config_logger, add_file_handler_to_root


LOG = config_logger(__name__)
add_file_handler_to_root('nn_run.log')

tmp_file = 'nn_last_tmp_input_pybrain.tmp'

output_names =[
'ager',
'tau_V',
'agem',
'tlastb',
'Mstars',
'xi_Wtot',
'sfr29',
'xi_PAHtot',
'f_muSFH',
'fb17',
'fb16',
'T_CISM',
'Ldust',
'mu_parameter',
'xi_Ctot',
'f_muIR',
'fb18',
'fb19',
'T_WBC',
'SFR_0_1Gyr',
'fb29',
'sfr17',
'sfr16',
'sfr19',
'sfr18',
'tau_VISM',
'sSFR_0_1Gyr',
'metalicity_Z_Z0',
'Mdust',
'xi_MIRtot',
'tform',
'gamma']

def normalise_2Darray(array, ignore=0):
    x_len = len(array)
    y_len = len(array[0])

    minimums = [0] * len(array[0])
    maximums = [0] * len(array[0])

    for y in range(0, y_len):
        minimum = array[y][0]
        maximum = array[y][0]

        for x in range(0, x_len):

            if ignore > 0:
                ignore -= 1
                continue

            if array[x][y] > maximum:
                maximum = array[x][y]
            elif array[x][y] < minimum:
                minimum = array[x][y]

        for x in range(0, x_len):
            array[x][y] = (array[x][y] - minimum) / float(maximum - minimum)

        minimums[y] = minimum
        maximums[y] = maximum

    return minimums, maximums, array


def denormalise_value(value, minimum, maximum):
    return value * (maximum - minimum) + minimum


def check_temp(filename, config):
    if os.path.exists(filename):
        # read in the header
        with open(filename) as f:
            header = pickle.load(f)

            for key in config:
                if header[key] != config[key]:
                    return False

            return True
    else:
        return False


def load_from_file(filename):
    with open(filename) as f:
        header = pickle.load(f)
        test_in = pickle.load(f)
        test_out = pickle.load(f)
        train_in = pickle.load(f)
        train_out = pickle.load(f)
        galaxy_ids = pickle.load(f)


    return test_in, test_out, train_in, train_out, galaxy_ids


def write_file(filename, config, test_in, test_out, train_in, train_out, galaxy_ids):
    with open(filename, 'w') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_in, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_out, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_in, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_out, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(galaxy_ids, f, pickle.HIGHEST_PROTOCOL)


# Inputs: 40 parameters + repeat_redshift * redshift values.
# Repeat redshift can be used to add redshift in to the input layer multiple times.

# Outputs:
# median = 32
# best_fit = 16
# best_fit_model = 4
# best_fit_inputs = 20. That weird line in the fit file that contains different values for the standard inputs.
db_init('sqlite:///Database_run06.db')
train_data = 10000
test_data = 100
run_id = '06'
output_type = 'median'
input_type = 'normal'
repeat_redshift = 1


def run_network(connections, layers, single_value=None, input_filter_types=None):

    nn_config_dict = {'test':test_data, 'train':train_data, 'run':run_id, 'input_type': input_type, 'output_type':output_type, 'repeat_redshift':repeat_redshift, 'value':single_value, 'input_filter_types':input_filter_types}

    if check_temp(tmp_file, nn_config_dict):
        LOG.info('Correct temp file exists at {0}, loading from temp'.format(tmp_file))
        test_in, test_out, train_in, train_out, galaxy_ids = load_from_file(tmp_file)
        LOG.info('Done.')
    else:
        LOG.info('No temp file, reading from database.')
        test_in, test_out, train_in, train_out, galaxy_ids = get_train_test_data(test_data, train_data, input_type=input_type, output_type=output_type, repeat_redshift=repeat_redshift, single_value=single_value, input_filter_types=input_filter_types)

        LOG.info('Done. Writing temp file for next time.')
        write_file(tmp_file, nn_config_dict, test_in, test_out, train_in, train_out, galaxy_ids)
        LOG.info('Done. Temp file written to {0}'.format(tmp_file))

    LOG.info('\nNormalising...')
    train_in_min, train_in_max, train_in = normalise_2Darray(train_in)
    #train_out_min, train_out_max, train_out = normalise_2Darray(train_out)

    test_in_min, test_in_max, test_in = normalise_2Darray(test_in)
    #test_out_min, test_out_max, test_out = normalise_2Darray(test_out)

    LOG.info('Normalising done.')

    print np.shape(train_in)
    print np.shape(train_out)
    print np.shape(test_in)
    print np.shape(test_out)

    input_dim = 0

    if 'optical' in input_filter_types:
        input_dim += 10

    if 'ir' in input_filter_types:
        input_dim += 9

    if 'uv' in input_filter_types:
        input_dim += 2

    input_dim *= 2

    data_set = SupervisedDataSet(input_dim+repeat_redshift, 15)

    for i in range(0, len(train_in)):
        data_set.addSample(train_in[i], train_out[i])

    LOG.info('Compiling neural network model')

    network = FeedForwardNetwork()

    input_layer = TanhLayer(input_dim+repeat_redshift,'Input')
    network.addInputModule(input_layer)

    prev_layer = TanhLayer(connections, 'hidden0')
    network.addModule(prev_layer)
    network.addConnection(FullConnection(input_layer, prev_layer))

    for i in range(0, layers):
        new_layer = TanhLayer(connections, 'hidden{0}'.format(i))
        network.addModule(new_layer)
        network.addConnection(FullConnection(new_layer, prev_layer))
        prev_layer = new_layer

    output_layer = LinearLayer(15, 'output')
    network.addOutputModule(output_layer)
    network.addConnection(FullConnection(new_layer, output_layer))

    network.sortModules()

    trainer = BackpropTrainer(network, data_set, verbose=True)

    LOG.info("Compiled.")

    epochs = 0
    do_test = 10
    trained = False
    while not trained:
        error = trainer.train()
        epochs += 1
        do_test -= 1

        LOG.info('Error rate at epoch {0}: {1}'.format(epochs, error))

        if error < 0.001 or epochs == 500:
            trained = True

        if do_test == 0:
            with open('pybrain_inputs_{0}_outputs_{1}.txt'.format(input_filter_types, output_names[single_value]), 'w') as f:
                for i in range(0, 20):

                    test_to_use = rand.randint(0, test_data - 1)
                    ans = network.activate(np.array(test_in[test_to_use]))

                    #f.write('Test {0} for epoch {1}\n'.format(i, total_epoch))
                    f.write('\nGalaxy number = {0}\n'.format(galaxy_ids[test_to_use]))
                    f.write('Inputs: ')
                    for item in test_in[test_to_use]:
                        f.write(str(item))
                    f.write('\nOutput   Correct\n')
                    for a in range(0, len(test_out[test_to_use])):
                        #f.write('{0}: {1}  =   {2}\n'.format(output_names[a], denormalise_value(ans[a], train_out_min[a], train_out_max[a]), denormalise_value(test_out[test_to_use][a], test_out_min[a], test_out_max[a])))
                        f.write('{0}: {1}  =   {2}\n'.format(output_names[a], ans[a], test_out[test_to_use][a]))
                    f.write('\n\n')

            do_test = 10

if __name__ == '__main__':
    filters = ['ir', 'uv', 'optical']
    parameters = range(0, 15)

    #for filter in filters:
    for parameter in parameters:
        run_network(input_filter_types=['ir', 'uv', 'optical'], single_value=parameter)

LOG.info("Done")



