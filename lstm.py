from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Activation

from config import embedding_vector_length, num_classes, hidden_size, vocab_size, max_sentence_lenght


def create_model():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sentence_lenght))
    model.add(LSTM(hidden_size))
    model.add(Dropout('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
