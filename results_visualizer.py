import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import pandas

plt.close('all')


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

def plot_grouped(df, group_by, x, y, title):

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df in df.groupby(group_by):
        df.plot(x=x, y=y, ax=ax, label=group_by + ' ' + str(label), title=title)
    plt.legend()

nn_results = pandas.read_csv('results/nn_results.csv')
rnn_results = pandas.read_csv('results/rnn_results.csv')
lstm_results = pandas.read_csv('results/lstm_results.csv')

fig, ax = plt.subplots(figsize=(8,6))

plot_grouped(lstm_results, group_by='embedding_size', x='hidden_size', y='test_accuracy', title='a')
plot_grouped(lstm_results, group_by='hidden_size', x='embedding_size', y='test_accuracy', title='a')
plt.show()

