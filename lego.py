import numpy as np
import time
import sys
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from scipy.optimize import minimize, Bounds

from multistart import generate
from testFunctions import rosen, rastrigin, schwefel


def lego(f, threshold, clf, n_dimensions=2, maxRange=5.12, numSamples=100, numTrainingSamples=1000, visualize=True,
         validation=False):
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

    if validation:
        clf.C, clf.gamma, clf.class_weight, clf.kernel = validate(xTrain, yTrain, xTest, yTest)

    # training
    clf.fit(xTrain, yTrain.ravel())

    # testing
    predictions = clf.predict(xTest)
    right = 0
    positivesRight = 0
    for i in range(len(predictions)):
        if predictions[i] == yTest[i]:
            right += 1
        if predictions[i] == yTest[i] and predictions[i] == 1:
            positivesRight += 1

    print('Accuracy: ' + str((right / len(predictions)) * 100) + '%')
    print('Positives in test set: ' + str(len([x for x in yTest if x == 1])))
    print('Positive examples accuracy: ' + str((positivesRight / len([x for x in yTest if x == 1])) * 100) + '%')

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


def validate(xTrain, yTrain, xTest, yTest):
    parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}]}]

    print('Tuning hyper-parameters for recall')

    clf = GridSearchCV(SVC(), parameters, scoring='recall_macro')
    clf.fit(xTrain, yTrain.ravel())

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    yTrue, yPred = yTest, clf.predict(xTest)
    print(classification_report(yTrue, yPred))
    print()

    return clf.best_params_['C'], clf.best_params_['gamma'], clf.best_params_['class_weight'], clf.best_params_['kernel']



if __name__ == '__main__':

    C = 0.01
    gamma = 1e-6
    class_weight = {1: 50}

    clf = SVC(C=C, gamma=gamma, class_weight=class_weight)

    results = []
    numSamples = 1000
    numTrainingSamples = 10000
    # recommended values:
    # Rastrigin: 2D: 1, 3D: 5, 10D: 60
    threshold = -3000
    visualize = True
    validation = True
    nDimensions = 10
    maxRange = 500

    start = time.time()
    best, point, goodOptChance, numRuns, samples = lego(schwefel, threshold=threshold, clf=clf, numSamples=numSamples,
                                                        numTrainingSamples=numTrainingSamples, n_dimensions=nDimensions,
                                                        maxRange=maxRange, visualize=visualize, validation=validation)
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