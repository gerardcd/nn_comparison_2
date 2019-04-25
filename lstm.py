from __future__ import print_function

from keras.backend import dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM, Activation

import matplotlib.pyplot as plt

from config import batch_size, embedding_vector_length, num_classes, epochs, hidden_size, vocab_size, max_sentence_lenght


def createModel(xTrain, yTrain, xTest, yTest):

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_sentence_lenght))
    model.add(LSTM(hidden_size))
    model.add(Dropout('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    history = model.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, verbose=1)

    score = model.evaluate(xTest, yTest, batch_size=batch_size, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    model.save_weights("model.h5")
    print("Saved model to disk")

    plt.plot(history.history['loss'])
    plt.show()
