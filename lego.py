import numpy as np
import time

from sklearn.svm import SVC
from scipy.optimize import minimize

from multistart import generate
from testFunctions import rosen, rastrigin, schwefel


def lego(f, threshold, n_dimensions=2, maxRange=5.12, numSamples=100, numTrainingSamples=1000):
    actualBest = float('inf')
    bestPoint = []
    trainSetX = np.array([])
    trainSetY = np.array([])

    for i in range(numTrainingSamples):
        sample = generate(n_dimensions, maxRange)
        res = minimize(f, sample, method='nelder-mead', options={'xatol': 1e-8})

        trainSetX = np.append(trainSetX, [res.x])

        if res['fun'] < threshold:
            trainSetY = np.append(trainSetY, np.array(1))
        else:
            trainSetY = np.append(trainSetY, np.array(-1))

        if res['fun'] < actualBest:
            actualBest = res['fun']
            bestPoint = res.x

    trainSetX = trainSetX.reshape(numTrainingSamples, n_dimensions)
    trainSetY = trainSetY.reshape(-1, 1)

    clf = SVC(gamma='auto')
    clf.fit(trainSetX, trainSetY)

    for i in range(numSamples):
        sample = generate(n_dimensions, maxRange)
        if clf.predict([sample]) is 1:
            res = minimize(f, sample, method='nelder-mead', options={'xatol': 1e-8})
            if res['fun'] < actualBest:
                actualBest = res['fun']
                bestPoint = res.x

    return actualBest, bestPoint


start = time.time()
best, point = lego(rastrigin, threshold=10, n_dimensions=6, maxRange=5.12)
end = time.time()
print('best: ' + str(best) + '    point: ' + str(point) + '    time elapsed: ' + str(end - start))

