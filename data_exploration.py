import matplotlib.pyplot as plt
import numpy as np

from data import dataFromDB

xData, yData = dataFromDB()

lengths = [len(tweet) for tweet in xData]

plt.hist(lengths, bins='auto')
plt.title("Histogram of tweets length")
plt.show()

print "Tweets with 50 words or less: {}%".format(float(len(filter(lambda x: x <= 50, lengths))) / float(len(lengths)) * 100)

words = {}

for tweet in xData:
    for word in tweet:
        if not word in words.keys():
            words[word] = 1
        else:
            words[word] += 1

print "Number of different stemmed words: ", len(words.keys())

repetitions = words.values()
repetitions.sort(reverse=True)
acc_repetitions = np.cumsum(repetitions)

plt.plot(acc_repetitions)
plt.title("Accumulated word frequencies")
plt.xlabel("vocabulary size")
plt.ylabel("repetitions in tweet set")
plt.show()

acc_1000 = acc_repetitions[999]

print "Accumulated repetitions with 1000 words: {} ({}%)".format(acc_1000, (float(acc_1000 ) / float(acc_repetitions[-1])) * 100)