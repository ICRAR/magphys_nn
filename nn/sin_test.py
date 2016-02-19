import os, sys, time

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))

import random as rand
import math

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout
from keras.optimizers import *
import numpy as np
import matplotlib.pyplot as plt
from common.logger import config_logger, add_file_handler_to_root


LOG = config_logger(__name__)

x = np.arange(-math.pi, math.pi, 0.001)

y = []

for item in x:
    y.append([math.sin(item)])

x = [[i] for i in x]

print np.shape(x)
print np.shape(y)

for i in range(0, 5):
    print x[i][0], y[i][0]

x = np.array(x)
y = np.array(y)

optimiser = SGD(lr=0.01, momentum=0.3, decay=0.00000, nesterov=True)

model = Sequential()

model.add(Dense(output_dim=20, input_dim=1, init='glorot_uniform', activation='tanh'))

model.add(Dense(output_dim=1, input_dim=20, init='glorot_uniform', activation='linear'))
model.compile(loss='mse', optimizer=optimiser)

print 'Compiled'

trained = False
total_epoch = 0

while not trained:

    LOG.info('epoch {0} / {1}'.format(total_epoch, 5000))

    history = model.fit(x, y, batch_size=1000,
                        nb_epoch=1000,
                        validation_split=0.3,
                        show_accuracy=True, verbose=True)

    total_epoch += 1000

    if total_epoch >= 5000:
        trained = True


x = []
y = []
y2 = []
for i in np.arange(-math.pi * 2, math.pi * 2, 0.01):
    x.append(i)
    predict = model.predict(np.array([[i]]))
    y.append(predict[0][0])
    y2.append(math.sin(i))

plt.clf()
plt.plot(x, y, x, y2)
plt.savefig(filename='test.png', format='png')




