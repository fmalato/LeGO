import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize, Bounds

from multistart import generate
from testFunctions import rosen, rastrigin, schwefel


def lego(f, threshold, clf, n_dimensions=2, maxRange=5.12, numSamples=100, numTrainingSamples=1000, visualize=True):
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

    samples = []
    while len(stats) < numSamples:
        sample = generate(n_dimensions, maxRange)
        s = clf.predict([sample])
        if visualize:
            samples.append((sample, s[0]))
        if s[0] == 1.0:
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

    return actualBest, bestPoint, goodOptChance, numRuns, samples


clf = SVC(gamma='auto')
results = []
numSamples = 100
# recommended values:
# Rastrigin: 2D: 1, 3D: 5, 10D: 60
threshold = 1
visualize = True
nDimensions = 2

start = time.time()
best, point, goodOptChance, numRuns, samples = lego(rastrigin, threshold=threshold, clf=clf, numSamples=numSamples,
                                                    n_dimensions=nDimensions, maxRange=5.12, visualize=visualize)
results.append(best)
end = time.time()
print('best: ' + str(best) + '    point: ' + str(point) + '    time elapsed: ' + str(end - start))
print('An acceptable optimum was found ' + str(goodOptChance) + ' times out of ' + str(numSamples) + ' trials.')
print('NumRuns: ' + str(numRuns))

# 2D visualization
if visualize and nDimensions == 2:
    for i in range(len(samples)):
        if samples[i][1] == 1.0:
            plt.plot(samples[i][0][0], samples[i][0][1], 'bo')
        else:
            plt.plot(samples[i][0][0], samples[i][0][1], 'ro')

    plt.show()

# 3D visualization
if visualize and nDimensions == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, -45)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for i in range(len(samples)):
        if samples[i][1] == 1.0:
            ax.scatter(samples[i][0][0], samples[i][0][1], samples[i][0][2], marker='.', c='#0000ff')
        else:
            ax.scatter(samples[i][0][0], samples[i][0][1], samples[i][0][2], marker='.', c='#ff0000')
    plt.show()

