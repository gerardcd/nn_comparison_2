import time
import subprocess
import re
import operator

NN = 'NN'
RNN = 'RNN'
LSTM = 'LSTM'

TRAIN_LOSS = 'Train loss: '
TRAIN_ACC = 'Train accuracy: '

TEST_LOSS = 'Test loss: '
TEST_ACC = 'Test accuracy: '


# Extract accuracy value from the scripts results
def extraxtMetric(metric, output):
    regex = re.compile(metric + "[0-9]\.[0-9]+")
    search = regex.search(output)

    metric_str = search.group()
    metric = float(metric_str[len(metric):])

    return metric


# Run a model and return its metrics and running time
def runInstance(model, hidden_size, embbeding_size=None):
    command = 'python trainer.py'

    start = time.time()

    params = [model, hidden_size]
    if model in [RNN, LSTM]:
        params += [embbeding_size]

    params_str = ' '.join(map(str, params))
    command += ' ' + params_str

    print 'Running:', command

    output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()

    end = time.time()

    elapsed_time = end - start

    train_loss = extraxtMetric(TRAIN_LOSS, output)
    train_accuracy = extraxtMetric(TRAIN_ACC, output)

    test_loss = extraxtMetric(TEST_LOSS, output)
    test_accuracy = extraxtMetric(TEST_ACC, output)

    return train_loss, train_accuracy, test_loss, test_accuracy, elapsed_time


# Average instance metrics
def runInstanceAvg(*args):
    rounds = 1

    metrics_avg = (.0, .0, .0, .0, .0)
    for i in range(rounds):
        metrics = runInstance(*args)
        metrics_avg = map(operator.add, metrics, metrics_avg)

    metrics_avg = map(lambda metric: metric / float(rounds), metrics_avg)

    return metrics_avg


# TIME AND ACCURACY EXPERIMENTS

twos = [2 ** x for x in range(4, 10)]

# Empty the results files
open('results/nn_results.csv', 'w').close()
open('results/rnn_results.csv', 'w').close()
open('results/lstm_results.csv', 'w').close()

with open('results/nn_results.csv', 'a') as nn_results, open('results/rnn_results.csv', 'a') as rnn_results, open('results/lstm_results.csv', 'a') as lstm_results:

    nn_results.write('hidden_size, train_loss, train_accuracy, test_loss, test_accuracy, time\n')
    rnn_results.write('hidden_size, embedding_size, train_loss, train_accuracy, test_loss, test_accuracy, time\n')
    lstm_results.write('hidden_size, embedding_size, train_loss, train_accuracy, test_loss, test_accuracy, time\n')

    for hidden_size in twos:
        # Run NN
        result = runInstanceAvg(NN, hidden_size)
        result_str = ', '.join(map(str, [hidden_size] + result)) + '\n'
        nn_results.write(result_str)

        for embedding_size in twos:
            # Run RNN
            result = runInstanceAvg(RNN, hidden_size, embedding_size)
            result_str = ', '.join(map(str, [hidden_size, embedding_size] + result)) + '\n'
            rnn_results.write(result_str)

            # Run LSTM
            result = runInstanceAvg(LSTM, hidden_size, embedding_size)
            result_str = ', '.join(map(str, [hidden_size, embedding_size] + result)) + '\n'
            lstm_results.write(result_str)

