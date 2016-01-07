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

tmp_file = '/tmp/nn_last_tmp_input_pybrain.tmp'


def standardise_2Darray(array):

    x_len = len(array)
    y_len = len(array[0])

    means = [0] * len(array[0])
    ranges = [0] * len(array[0])

    for y in range(0, y_len):
        mean = 0
        largest = array[0][y]
        smallest = array[0][y]
        for x in range(0, x_len):
            mean += array[x][y]

            if array[x][y] > largest:
                largest = array[x][y]
            elif array[x][y] < smallest:
                smallest = array[x][y]

        mean /= float(x_len)

        ranged = largest - smallest

        for x in range(0, x_len):
            array[x][y] = (array[x][y] - mean) / float(ranged)

        means[y] = mean
        ranges[y] = ranged

    return means, ranges, array


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

            for key in header:
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


def run_network(single_value=None):

    nn_config_dict = {'test':test_data, 'train':train_data, 'run':run_id, 'input_type': input_type, 'output_type':output_type, 'repeat_redshift':repeat_redshift, 'value':single_value}

    if check_temp(tmp_file, nn_config_dict):
        print 'Correct temp file exists at {0}, loading from temp'.format(tmp_file)
        test_in, test_out, train_in, train_out, galaxy_ids = load_from_file(tmp_file)
        print 'Done.'
    else:
        print 'No temp file, reading from database.'
        test_in, test_out, train_in, train_out, galaxy_ids = get_train_test_data(test_data, train_data, run_id, input_type=input_type, output_type=output_type, repeat_redshift=repeat_redshift, single_value=single_value)

        print 'Done. Writing temp file for next time.'
        write_file(tmp_file, nn_config_dict, test_in, test_out, train_in, train_out, galaxy_ids)
        print 'Done. Temp file written to {0}'.format(tmp_file)

    """
    redshift_train = []
    redshift_test = []
    for i in range(0, len(train_in)):
        redshift_train.append([train_in[i][0]])

    for i in range(0, len(test_in)):
        redshift_test.append([test_in[i][0]])
    """
    #redshift_test = np.array(redshift_test)
    #redshift_train = np.array(redshift_train)

    print 'Normalising...'
    train_in_min, train_in_max, train_in = normalise_2Darray(train_in)
    train_out_min, train_out_max, train_out = normalise_2Darray(train_out)


    test_in_min, test_in_max, test_in = normalise_2Darray(test_in)
    test_out_min, test_out_max, test_out = normalise_2Darray(test_out)

    #redshift_train_min, redshift_train_max, redshift_train = normalise_2Darray(redshift_train)
    #redshift_test_min, redshift_test_max, redshift_test = normalise_2Darray(redshift_test)

    print 'Normalising done.'

    print np.shape(train_in)
    print np.shape(train_out)
    print np.shape(test_in)
    print np.shape(test_out)
    #print np.shape(redshift_train)

    data_set = SupervisedDataSet(40+repeat_redshift, 32)

    for i in range(0, len(train_in)):
        data_set.addSample(train_in[i], train_out[i])

    print 'Compiling neural network model'

    network = FeedForwardNetwork()

    input_layer = LinearLayer(40+repeat_redshift,'Input')
    hidden1 = TanhLayer(50,'hidden1')
    hidden2 = TanhLayer(50,'hidden2')
    hidden3 = TanhLayer(50,'hidden3')
    output_layer = LinearLayer(32, 'output')

    network.addInputModule(input_layer)
    network.addModule(hidden1)
    network.addModule(hidden2)
    network.addModule(hidden3)
    network.addOutputModule(output_layer)

    network.addConnection(FullConnection(input_layer, hidden1))
    network.addConnection(FullConnection(hidden1, hidden2))
    network.addConnection(FullConnection(hidden2, hidden3))
    network.addConnection(FullConnection(hidden3, output_layer))

    network.sortModules()

    trainer = BackpropTrainer(network, data_set)

    print "Compiled."

    epochs = 0
    do_test = 100
    trained = False
    while not trained:
        error = trainer.train()
        epochs += 1
        do_test -= 1

        print 'Error rate at epoch {0}: {1}'.format(epochs, error)

        if error < 0.001 or epochs == 1000:
            trained = True

        if do_test == 0:
            for i in range(0, 10):
                test_to_use = rand.randint(0, test_data - 1)
                ans = network.activate(np.array(test_in[test_to_use]))
                #ans = graph.predict({'input':np.array([test_in[test_to_use]]), 'redshift':np.array([redshift_test[test_to_use]])})
                #f.write('Test {0} for epoch {1}\n'.format(i, total_epoch))
                print '\nGalaxy number = {0}\n'.format(galaxy_ids[test_to_use])
                print 'Output   Correct\n'
                for a in range(0, len(test_out[test_to_use])):
                    print'{0}  =   {1}\n'.format(denormalise_value(ans[a], train_out_min[a], train_out_max[a]), denormalise_value(test_out[test_to_use][a], test_out_min[a], test_out_max[a]))
                print '\n\n'
            do_test = 100

if __name__ == '__main__':
    run_network()

print "Done"



