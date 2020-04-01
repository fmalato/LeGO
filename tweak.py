import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, Bounds

from multistart import generate
from testFunctions import rosen, rastrigin, schwefel

def tweaking(f, clf, maxRange, threshold, numTrainingSamples=1000, n_dimensions=2):
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
    clf.fit(xTrain, yTrain.ravel())

    # testing
    predictions = clf.predict(xTest)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == yTest[i]:
            right += 1

    predPos = [p for p in predictions if p == 1.0]
    yTestPos = [t for t in yTest if t == 1.0]
    accPos = (len(predPos) / len(yTestPos)) * 100

    return accPos


gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
bestC = float('inf')
bestGamma = 0.0
bestAcc = 0.0

with open('results-stored.txt', 'w+') as f:
    thresholds = [-700, -900, -1100, -1300, -1500, -1700, -1900, -2100, -2200]
    for dim in range(2, 11):
        bestC = float('inf')
        bestGamma = 0.0
        bestAcc = 0.0
        for gamma in gammas:
            for i in range(100):
                C = 0.01 * (i + 1)
                print('Actual: C = {C}, gamma = {g}'.format(C=C, g=gamma))
                clf = SVC(C=C, gamma=gamma)
                accPos = tweaking(schwefel, clf, 500, thresholds[dim - 2], 1000, dim)
                if accPos > bestAcc:
                    bestAcc = accPos
                    print('New Best: {a}%'.format(a=bestAcc))
                    bestC = C
                    bestGamma = gamma

        print('Dimension: {d}D'.format(d=dim))
        print('Best configuration: C = {C}, gamma = {g}'.format(C=bestC, g=bestGamma))
        f.write('Dimension: {d}D\n'.format(d=dim))
        f.write('Best configuration: C = {C}, gamma = {g}\n'.format(C=bestC, g=bestGamma))
