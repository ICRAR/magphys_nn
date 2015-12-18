from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import History
import numpy as np
import random as rand

#(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.pkl",
#                                                     nb_words=None, skip_top=0, maxlen=None, test_split=0.1, seed=113)

#print X_train
#print y_train

print "Simple KERAS XOR example"
#X = numpy.array([[2, 2, 1, 0], [2, 3, 1, 0]]), y = [4, 5]
train_in = np.array([[0,0], [1,0], [0,1], [1,1]])
#train_out = [0, 1, 1, 0]
train_out = np.array([0, 1, 1, 0])

model = Sequential()  # Use sequential network (input -> hidden -> output)
print 1

# 3 Layers: 2node input, 2 node hidden, 1 node output
model.add(Dense(output_dim=3, input_dim=2, init='uniform'))
model.add(Activation('tanh'))
model.add(Dense(output_dim=3, input_dim=3, activation='tanh'))
model.add(Dense(output_dim=1, input_dim=3, activation='tanh'))

model.compile(loss='mse', optimizer='sgd')

print "Compiled"

# Time to train

#model.fit(train_in, train_out, batch_size=16, nb_epoch=1, show_accuracy=True, verbose=True)

# Train the model each generation and show predictions against the validation dataset
history = History()
trained = False
total_epoch = 0
while not trained:
    #print '\n'
    #print('-' * 50)
    #print('Iteration', iteration)
    history = model.fit(train_in, train_out, batch_size=400, nb_epoch=25000, show_accuracy=False, verbose=False, callbacks=[history])
    current_loss = history.totals['loss']

    total_epoch += 25000
    print history.totals
    if history.totals['loss'] < 0.0000001 or total_epoch > 1000000:
        trained = True

 ###
 # Select 10 samples from the validation set at random so we can visualize errors
for i in range(10):
    inp = list()
    inp.append(rand.randint(0, 1))
    inp.append(rand.randint(0, 1))
    result = inp[0] ^ inp[1]
    npinp = np.array([inp])

    ans = model.predict(npinp)

    print inp[0], inp[1], result
    print '{0} = {1}'.format(ans, result)
    if ans - result < 0.001:
        print 'Correct!\n'
    else:
        print 'Incorrect!\n'

print "Done"



