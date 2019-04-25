import sys
import lstm
import nn
import rnn
import matplotlib.pyplot as plt

from data import getData
from config import batch_size, epochs, verbose

net = sys.argv[1]
hidden_size = int(sys.argv[2])
embedding_vector_length = None

if net in ['LSTM', 'RNN']:
    embedding_vector_length = int(sys.argv[3])

(xTrain, yTrain, xTest, yTest) = getData(net)

if net == 'NN':
    model = nn.create_model(hidden_size)

elif net == 'LSTM':
    model = lstm.create_model(hidden_size, embedding_vector_length)

elif net == 'RNN':
    model = rnn.create_model(hidden_size, embedding_vector_length)

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