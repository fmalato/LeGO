import numpy as np
import time

from sklearn.svm import SVC
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
        res = minimize(f, sample, method='nelder-mead', options={'xatol': 1e-8})

        trainSetX = np.append(trainSetX, sample)

        if res['fun'] < threshold:
            trainSetY = np.append(trainSetY, np.array(1))
        else:
            trainSetY = np.append(trainSetY, np.array(-1))

        if res['fun'] < actualBest:
            actualBest = res['fun']
            bestPoint = res.x

    # stats
    positives = 0
    for i in range(len(trainSetY)):
        if trainSetY[i] == np.array(1):
            positives += 1
    print('Positive examples: ' + str(positives) + '/1000')

    trainSetX = trainSetX.reshape(numTrainingSamples, n_dimensions)
    trainSetY = trainSetY.reshape(-1, 1)

    clf.fit(trainSetX, trainSetY)

    for i in range(numSamples):
        sample = generate(n_dimensions, maxRange)
        if clf.predict([sample]) is 1:
            res = minimize(f, sample, method='nelder-mead', options={'xatol': 1e-8})
            if res['fun'] < actualBest:
                actualBest = res['fun']
                bestPoint = res.x

    return actualBest, bestPoint


clf = SVC(gamma='auto')
results = []
for i in range(300):
    print('Iteration ' + str(i))
    start = time.time()
    best, point = lego(rastrigin, threshold=2, clf=clf, n_dimensions=2, maxRange=5.12)
    results.append(best)
    end = time.time()
    print('best: ' + str(best) + '    point: ' + str(point) + '    time elapsed: ' + str(end - start))

globalOptChance = 0
for i in range(len(results)):
    if results[i] == 0.0:
        globalOptChance += 1

print('Global optimum was found ' + str(globalOptChance) + ' times out of 300 trials.')


