from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN, Activation

from config import num_classes, vocab_size, max_sentence_lenght


def create_model(hidden_size, embedding_vector_length):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sentence_lenght))
    model.add(SimpleRNN(hidden_size))
    model.add(Dropout('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
