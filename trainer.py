import sys

net = sys.argv[1] or 'NN'

from data import getData
import nn, lstm

(xTrain, yTrain, xTest, yTest) = getData(net)

if net == 'NN':
    nn.createModel(xTrain, yTrain, xTest, yTest)

if net == 'LSTM':
    lstm.createModel(xTrain, yTrain, xTest, yTest)

exit(0)