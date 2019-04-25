import sys
import lstm
import nn
import rnn
import matplotlib.pyplot as plt

from data import getData
from config import batch_size, epochs, verbose

net = sys.argv[1] or 'NN'

(xTrain, yTrain, xTest, yTest) = getData(net)

if net == 'NN':
    model = nn.create_model()

elif net == 'LSTM':
    model = lstm.create_model()

elif net == 'RNN':
    model = rnn.create_model()

else:
    raise Exception('Unknown model specified')

print(model.summary())

history = model.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, verbose=verbose)

print 'Train loss:', history.history['loss'][-1]
print 'Train accuracy:', history.history['acc'][-1]

score = model.evaluate(xTest, yTest, batch_size=batch_size, verbose=verbose)

print 'Test loss:', score[0]
print 'Test accuracy:', score[1]


plt.plot(history.history['loss'])
plt.show()