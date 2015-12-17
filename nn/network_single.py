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
from keras.layers.core import Dense, Activation
from keras.callbacks import History
from keras.optimizers import SGD
from keras.layers.advanced_activations import ELU
import numpy as np
from common.database import get_train_test_data


print "Simple KERAS XOR example"

# Value from 0 - 31
for field in range(0, 1):
    test_in, test_out, train_in, train_out = get_train_test_data(500, 10000, '06', single_value=field)

    print np.shape(test_in)
    print np.shape(test_out)

    print np.shape(train_in)
    print np.shape(train_out)

    model = Sequential()
    optimiser = SGD(lr=0.001, momentum=0.1, decay=0, nesterov=True)

    model.add(Dense(output_dim=80, input_dim=40, init='uniform', activation='sigmoid'))
    model.add(Dense(output_dim=1, input_dim=80, init='uniform', activation='sigmoid'))

    model.compile(loss='mape', optimizer=optimiser)

    print "Compiled"

    # Train the model each generation and show predictions against the validation dataset
    history = History()
    trained = False
    total_epoch = 0
    while not trained:

        history = model.fit(train_in, train_out, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=True, callbacks=[history])
        current_loss = history.totals['loss']

        total_epoch += 3
        print history.totals
        if history.totals['loss'] < 0.001 or total_epoch > 3:
            trained = True

    for i in range(0, 10):
        ans = model.predict(np.array([test_in[i]]))
        print 'Test {0} for epoch {1}'.format(i, total_epoch)
        print 'Output   Correct'
        print '{0}  =   {1}'.format(ans[0][0], test_out[i][0])
        print
        print

    model.save_weights('/home/ict310/keras_model{0}.hdf5'.format(field))

print "Done"



