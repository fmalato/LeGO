import time
import json
import numpy as np

from lego import lego
from sklearn.svm import SVC
from scipy.optimize import minimize
from testFunctions import rastrigin, schwefel
from multistart import multistartNew

# Rastrigin: 10000, 10000, 40, True, True, 9, 5.12, 0.01, 1e-6, 50
numSamples = 5
numTrainingSamples = 1000
threshold = 40
visualize = True
validation = True
nDimensions = 9
maxRange = 5.12

C = 0.01
gamma = 1e-6
class_weight = {1: 50}

clf = SVC(C=C, gamma=gamma, class_weight=class_weight)

start = time.time()
best, point, goodOptChance, numRuns, samples = lego(rastrigin, threshold=threshold, clf=clf, numSamples=numSamples,
                                                    numTrainingSamples=numTrainingSamples, n_dimensions=nDimensions,
                                                    maxRange=maxRange, visualize=visualize, validation=validation)

json_data = {}

json_data["lego"] = {}
json_data["multistart"] = {}
idx = 0
for sample in samples:
    point, value = minimize(rastrigin, sample[0], method='L-BFGS-B', options={'ftol': 1e-8})
    json_data["lego"][str(idx)].append({"point": point,
                                      "value": value})
    idx += 1
for i in range(len(samples)):
    value, point = multistartNew(rastrigin, nDimensions, maxRange, numSamples)
    json_data["multistart"][str(i)].append({"point": point,
                                            "value": value})
json.dump(json_data, "rastrigin.json")




