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

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation
from keras.callbacks import History
from keras.optimizers import *
import numpy as np
from common.database import get_train_test_data
import pickle
from keras.utils.visualize_util import to_graph

tmp_file = '/tmp/nn_last_tmp_input.tmp'


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
train_data = 50000
test_data = 500
run_id = '2015_01'
output_type = 'median'
repeat_redshift = 1

nn_config_dict = {'test':test_data, 'train':train_data, 'run':run_id, 'output_type':output_type, 'repeat_redshift':repeat_redshift}

if check_temp(tmp_file, nn_config_dict):
    print 'Correct temp file exists at {0}, loading from temp'.format(tmp_file)
    test_in, test_out, train_in, train_out, galaxy_ids = load_from_file(tmp_file)
    print 'Done.'
else:
    print 'No temp file, reading from database.'
    test_in, test_out, train_in, train_out, galaxy_ids = get_train_test_data(test_data, train_data, run_id, output_type=output_type, repeat_redshift=repeat_redshift)

    print 'Done. Writing temp file for next time.'
    write_file(tmp_file, nn_config_dict, test_in, test_out, train_in, train_out, galaxy_ids)
    print 'Done. Temp file written to {0}'.format(tmp_file)

redshift_train = []
redshift_test = []
for i in range(0, len(train_in)):
    redshift_train.append([train_in[i][0]])

for i in range(0, len(test_in)):
    redshift_test.append([test_in[i][0]])

redshift_test = np.array(redshift_test)
redshift_train = np.array(redshift_train)

print 'Normalising...'
train_in_min, train_in_max, train_in = normalise_2Darray(train_in)
train_out_min, train_out_max, train_out = normalise_2Darray(train_out)


test_in_min, test_in_max, test_in = normalise_2Darray(test_in)
test_out_min, test_out_max, test_out = normalise_2Darray(test_out)

redshift_train_min, redshift_train_max, redshift_train = normalise_2Darray(redshift_train)
redshift_test_min, redshift_test_max, redshift_test = normalise_2Darray(redshift_test)

print 'Normalising done.'

print np.shape(train_in)
print np.shape(train_out)
print np.shape(test_in)
print np.shape(test_out)
print np.shape(redshift_train)

print 'Compiling neural network model'

graph = Graph()

graph.add_input(name='input', input_shape=(41,))
graph.add_input(name='redshift', input_shape=(1,))
graph.add_node(Dense(60, input_dim=41, activation='tanh'), name='input_layer', input='input')
graph.add_node(Dense(60, activation='tanh'), name='hidden2', inputs=['input_layer', 'redshift'], merge_mode='concat')
graph.add_node(Dense(60, activation='tanh'), name='hidden3', input='hidden2')
graph.add_node(Dense(32, activation='tanh'), name='hidden4', input='hidden3')
graph.add_output(name='output', input='hidden4')

to_graph(graph).write_svg('graph1.svg')

#model = Sequential()
optimiser = SGD(lr=0.01, momentum=0.0, decay=0, nesterov=True)
graph.compile(loss={'output':'rmse'}, optimizer=optimiser)
#optimiser = Adagrad(lr=0.01, epsilon=1e-06)
#optimiser = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

"""
model.add(Dense(output_dim=60, input_dim=40+repeat_redshift, init='he_normal', activation='tanh'))
model.add(Dense(output_dim=60, input_dim=60, init='he_normal', activation='tanh'))
model.add(Dense(output_dim=60, input_dim=60, init='he_normal', activation='tanh'))
model.add(Dense(output_dim=60, input_dim=60, init='he_normal', activation='relu'))
model.add(Dense(output_dim=32, input_dim=60, init='he_normal', activation='linear'))
"""
"""
model.add(Dense(output_dim=100, input_dim=40+repeat_redshift, init='glorot_uniform', activation='tanh'))
model.add(Dense(output_dim=100, input_dim=100, init='glorot_uniform', activation='tanh'))
model.add(Dense(output_dim=100, input_dim=100, init='glorot_uniform', activation='tanh'))
model.add(Dense(output_dim=100, input_dim=100, init='glorot_uniform', activation='tanh'))
model.add(Dense(output_dim=100, input_dim=100, init='glorot_uniform', activation='tanh'))
model.add(Dense(output_dim=32, input_dim=100, init='glorot_uniform', activation='tanh'))
model.compile(loss='rmse', optimizer=optimiser)
"""

print "Compiled."

# Train the model each generation and show predictions against the validation dataset
history = History()
trained = False
total_epoch = 0
while not trained:

    history = graph.fit({'input':train_in, 'redshift':redshift_train, 'output':train_out}, batch_size=500, nb_epoch=50, validation_split=0.1, verbose=True, callbacks=[history])
    #history = model.fit(train_in, train_out, batch_size=500, nb_epoch=50, validation_split=0.1, show_accuracy=True, verbose=True, callbacks=[history])
    current_loss = history.totals['loss']

    total_epoch += 1
    print history.totals
    if history.totals['loss'] < 0.001 or total_epoch > 50:
        trained = True

    for i in range(0, 10):
        test_to_use = rand.randint(0, test_data - 1)
        #ans = model.predict(np.array([test_in[test_to_use]]))
        ans = graph.predict({'input':np.array([test_in[test_to_use]]), 'redshift':np.array([redshift_test[test_to_use]])})
        print 'Test {0} for epoch {1}'.format(i, total_epoch)
        print 'Galaxy number = {0}'.format(galaxy_ids[test_to_use])
        print 'Output   Correct'
        ans = ans['output']
        for a in range(0, len(test_out[test_to_use])):
            print '{0}  =   {1}'.format(denormalise_value(ans[0][a], train_out_min[a], train_out_max[a]), denormalise_value(test_out[test_to_use][a], test_out_min[a], test_out_max[a]))
        print
        print

print "Done"



