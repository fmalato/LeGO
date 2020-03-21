import numpy as np
import time

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

from multistart import generate
from testFunctions import rosen, rastrigin, schwefel


def lego(f, threshold, clf, n_dimensions=2, maxRange=5.12, numSamples=100, numTrainingSamples=1000):
    actualBest = float('inf')
    bestPoint = []
    trainSetX = np.array([])
    trainSetY = np.array([])

    for i in range(numTrainingSamples):
        sample = generate(n_dimensions, maxRange)
        res = minimize(f, sample, method='L-BFGS-B', options={'ftol': 1e-8})

        trainSetX = np.append(trainSetX, sample)

        if res['fun'] < threshold:
            trainSetY = np.append(trainSetY, np.array(1))
        else:
            trainSetY = np.append(trainSetY, np.array(-1))

        if res['fun'] < actualBest:
            actualBest = res['fun']
            bestPoint = res.x

    print('Pre training best: ' + str(actualBest))

    trainSetX = trainSetX.reshape(numTrainingSamples, n_dimensions)
    trainSetY = trainSetY.reshape(-1, 1)

    xTrain, xTest, yTrain, yTest = train_test_split(trainSetX, trainSetY, test_size=0.25)

    # stats
    positives = 0
    for i in range(len(yTrain)):
        if yTrain[i] == np.array(1):
            positives += 1
    print('Positive examples: ' + str(positives) + '/' + str(len(yTrain)))

    # training
    clf.fit(xTrain, yTrain)

    # validating
    predictions = clf.predict(xTest)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == yTest[i]:
            right += 1
    print('Accuracy: ' + str((right / len(predictions)) * 100) + '%')

    stats = []
    numRuns = 0
    print('Progress: 0/' + str(numSamples))
    while len(stats) < numSamples:
        sample = generate(n_dimensions, maxRange)
        if clf.predict([sample])[0] == 1.0:
            res = minimize(f, sample, method='L-BFGS-B', options={'ftol': 1e-8})
            if res['fun'] < actualBest:
                actualBest = res['fun']
                bestPoint = res.x
            stats.append(res['fun'])
            if len(stats) % 20 == 0:
                print('Progress: ' + str(len(stats)) + '/' + str(numSamples))
        numRuns += 1

    goodOptChance = 0
    for i in range(len(stats)):
        if stats[i] < threshold:
            goodOptChance += 1

    return actualBest, bestPoint, goodOptChance, numRuns


if __name__ == '__main__':

    clf = SVC(gamma='auto')
    results = []
    numSamples = 2000
    numTrainingSamples = 32000
    threshold = 50
    n_dimensions = 10

    start = time.time()
    best, point, goodOptChance, numRuns = lego(rastrigin, threshold=threshold, clf=clf, numSamples=numSamples,
                                               numTrainingSamples=numTrainingSamples, n_dimensions=n_dimensions,
                                               maxRange=5.12)
    results.append(best)
    end = time.time()

    print('best: ' + str(best) + '    point: ' + str(point) + '    time elapsed: ' + str(end - start))
    print('An acceptable optimum was found ' + str(goodOptChance) + ' times out of ' + str(numRuns) + ' trials.')


'''

Multistart dim:10 34000

best: 8.95462647602268    point: [-1.98991225e+00 -7.43738143e-08 -9.94958615e-01  9.94958606e-01
 -9.94958641e-01 -2.81956823e-09  9.94958636e-01  9.94958622e-01
 -1.66350992e-08 -5.27418536e-08]    time elapsed: 162.63828134536743


Lego dim:10 32000 + 2000

pretraining best: 11.9394986095372
Positive examples: 1512/24000
Accuracy: 96.72500000000001%
best: 5.969754342559838    point: [ 9.94958630e-01 -9.94958637e-01 -9.94958640e-01 -6.39987000e-09
  9.94958638e-01 -7.77808983e-09 -9.32782112e-09 -9.94958647e-01
 -9.94958641e-01 -9.48547615e-09]    time elapsed: 178.54057955741882
An acceptable optimum was found 1895 times out of 55779 trials.

'''