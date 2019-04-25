from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from config import num_classes, vocab_size


def create_model(hidden_size):

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(vocab_size,)))
    model.add(Dropout('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
