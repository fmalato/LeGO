import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from scipy.optimize import minimize, Bounds

from multistart import generate
from testFunctions import rosen, rastrigin, schwefel


def lego(f, threshold, clf, n_dimensions=2, maxRange=5.12, numSamples=100, numTrainingSamples=1000, visualize=True,
         validation=False, score='recall'):
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
        clf.C, clf.gamma, clf.class_weight, clf.kernel = validate(xTrain, yTrain, xTest, yTest, score)

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

    samples = []
    while len(stats) < numSamples:
        sample = generate(n_dimensions, maxRange)
        s = clf.predict([sample])
        if s[0] == 1.0:
            res = minimize(f, sample, method='L-BFGS-B', options={'ftol': 1e-8})
            if visualize:
                samples.append((sample, s[0]))
            if res['fun'] < actualBest:
                actualBest = res['fun']
                bestPoint = res.x
            stats.append(res['fun'])
            sys.stdout.write('\r Progress: {n}/{t}'.format(n=len(stats), t=numSamples))
            sys.stdout.flush()
        numRuns += 1

    goodOptChance = 0
    for i in range(len(stats)):
        if stats[i] < threshold:
            goodOptChance += 1
    print('\n')
    return actualBest, bestPoint, goodOptChance, numRuns, samples, clf.C, clf.gamma, clf.class_weight, clf.kernel


def validate(xTrain, yTrain, xTest, yTest, score='recall'):
    parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'class_weight': [{1: 1}, {1: 2}, {1: 5}, {1: 10}]}]

    print('Tuning hyper-parameters for %s' % score)

    clf = GridSearchCV(SVC(), parameters, scoring='%s_macro' % score)
    clf.fit(xTrain, yTrain.ravel())

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()

    print("Detailed classification report:")
    print()
    yTrue, yPred = yTest, clf.predict(xTest)
    print(classification_report(yTrue, yPred))
    print()

    return clf.best_params_['C'], clf.best_params_['gamma'], clf.best_params_['class_weight'], clf.best_params_['kernel']



if __name__ == '__main__':

    name = "schwefel"
    results = []
    if name == "schwefel":
        numSamples = 1000
        numTrainingSamples = 10000
        # recommended values:
        # Rastrigin: 2D: 1, 3D: 5, 10D: 60
        threshold = -3000
        visualize = True
        validation = True
        nDimensions = 10
        maxRange = 500
    else:
        numSamples = 100
        numTrainingSamples = 1000
        # recommended values:
        # Rastrigin: 2D: 1, 3D: 4, 10D: 60
        threshold = 4
        visualize = True
        validation = True
        nDimensions = 3
        maxRange = 5.12
        # Best values:
        # 2D: C=1.0, gamma='auto'
        # 3D: C=0.67, gamma=0.1
        C = 0.67
        gamma = 0.1

    C = 0.01
    gamma = 1e-6
    class_weight = {1: 50}

    clf = SVC(C=C, gamma=gamma, class_weight=class_weight)

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
        plt.xlim(xmin=-maxRange, xmax=maxRange)
        plt.ylim(ymin=-maxRange, ymax=maxRange)
        plt.show()

    # 3D visualization
    if visualize and nDimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, -45)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-maxRange, maxRange])
        ax.set_ylim([-maxRange, maxRange])
        ax.set_zlim([-maxRange, maxRange])
        for i in range(len(samples)):
            if samples[i][1] == 1.0:
                ax.scatter(samples[i][0][0], samples[i][0][1], samples[i][0][2], marker='.', c='#0000ff')
            else:
                ax.scatter(samples[i][0][0], samples[i][0][1], samples[i][0][2], marker='.', c='#ff0000')
        plt.show()
