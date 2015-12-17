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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import *
from keras.callbacks import History
from keras.optimizers import SGD
import numpy as np
from common.database import get_train_test_data


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


def normalise_2Darray(array):
    x_len = len(array)
    y_len = len(array[0])

    minimums = [0] * len(array[0])
    maximums = [0] * len(array[0])

    for y in range(0, y_len):
        minimum = array[y][0]
        maximum = array[y][0]

        for x in range(0, x_len):
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

test_in, test_out, train_in, train_out = get_train_test_data(500, 10000, '06')

train_in_min, train_in_max, train_in = normalise_2Darray(train_in)
train_out_min, train_out_max, train_out = normalise_2Darray(train_out)

test_in_min, test_in_max, test_in = normalise_2Darray(test_in)
test_out_min, test_out_max, test_out = normalise_2Darray(test_out)

"""
train_in_m, train_in_r, train_in = standardise_2Darray(train_in)
train_out_m, train_out_r, train_out = standardise_2Darray(train_out)

test_in_m, test_in_r, test_in = standardise_2Darray(test_in)
test_out_m, test_out_r, test_out = standardise_2Darray(test_out)

print train_in[0][0]
print train_out[0][0]
print test_in[0][0]
print test_out[0][0]
print
print test_in_r[0]
print test_out_r[0]
print train_in_r[0]
print train_out_r[0]
print
print test_in_m[0]
print test_out_m[0]
print train_in_m[0]
print train_out_m[0]
"""

model = Sequential()
optimiser = SGD(lr=0.01, momentum=0.0, decay=0, nesterov=True)

model.add(Dense(output_dim=40, input_dim=40, init='uniform', activation='sigmoid'))
model.add(Dense(output_dim=40, input_dim=40, init='uniform', activation='sigmoid'))
model.add(Dense(output_dim=40, input_dim=40, init='uniform', activation='sigmoid'))
model.add(Dense(output_dim=32, input_dim=40, init='uniform', activation='linear'))

model.compile(loss='mse', optimizer=optimiser)

print "Compiled"

# Train the model each generation and show predictions against the validation dataset
history = History()
trained = False
total_epoch = 0
while not trained:

    history = model.fit(train_in, train_out, batch_size=500, nb_epoch=100, show_accuracy=True, verbose=True, callbacks=[history])
    current_loss = history.totals['loss']

    total_epoch += 1
    print history.totals
    if history.totals['loss'] < 0.001 or total_epoch > 100:
        trained = True

    for i in range(0, 10):
        test_to_use = rand.randint(0, 499)
        ans = model.predict(np.array([test_in[test_to_use]]))
        print 'Test {0} for epoch {1}'.format(i, total_epoch)
        print 'Output   Correct'
        for a in range(0, 32):
            print '{0}  =   {1}'.format(denormalise_value(ans[0][a], train_out_min[a], train_out_max[a]), denormalise_value(test_out[test_to_use][a], test_out_min[a], test_out_max[a]))
        print
        print

print "Done"



