from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing import sequence

import sqlite3
import random
import re
import pickle

from nltk import word_tokenize
from nltk.stem import SnowballStemmer

from config import vocab_size, max_sentence_lenght

stemmer = SnowballStemmer('spanish')


def stem(text):
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r"(#\S+|http(s|)://\S+|@\S+)", "", text)

    return [stemmer.stem(word) for word in word_tokenize(text)]
    #return [word for word in text.split()]


def dataFromDB():
    conn = sqlite3.connect('tweets.sqlite')

    tweets1 = conn.execute('SELECT text, clas FROM tweets WHERE clas = 1').fetchall()
    tweets2 = conn.execute('SELECT text, clas FROM tweets WHERE clas = 2').fetchall()

    tweets = tweets1 + tweets2
    random.shuffle(tweets)

    xData = []
    yData = []

    for tweet in tweets:
        text = stem(tweet[0])
        clas = tweet[1] - 1

        xData.append(text)
        yData.append(clas)

    return (xData, yData)

def saveTokenizer(tokenizer):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved tokenizer to disk")


def getData(net):
    (x, y) = dataFromDB()

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x)

    saveTokenizer(tokenizer)

    x = tokenizer.texts_to_sequences(x)

    if net == 'NN':
        x = tokenizer.sequences_to_matrix(x, mode='binary')

    if net == 'LSTM':
        x = sequence.pad_sequences(x, maxlen=max_sentence_lenght)

    y = to_categorical(y, 2)

    n = len(x)
    nTrain = int(round(0.8 * n))

    xTrain = x[:nTrain]
    xTest = x[nTrain:]

    yTrain = y[:nTrain]
    yTest = y[nTrain:]

    return (xTrain, yTrain, xTest, yTest)