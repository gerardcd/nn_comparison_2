import matplotlib.pyplot as plt
import pandas

NN = 'NN'
RNN = 'RNN'
LSTM = 'LSTM'

HIDDEN_SIZE = 0
EMBEDDING_SIZE = 1

TRAIN_LOSS = -5
TRAIN_ACC = -4
TEST_LOSS = -3
TEST_ACC = -2
TIME = -1

nn_results = pandas.read_csv('results_bu/nn_results.csv')
rnn_results = pandas.read_csv('results_bu/rnn_results.csv')
lstm_results = pandas.read_csv('results_bu/lstm_results.csv')
