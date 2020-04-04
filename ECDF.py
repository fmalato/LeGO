import time, sys
import json
import numpy as np

from lego import lego
from sklearn.svm import SVC
from scipy.optimize import minimize
from testFunctions import rastrigin, schwefel
from multistart import multistartNew, generate

# Rastrigin: 10000, 10000, 40, True, True, 9, 5.12, 0.01, 1e-6, 50
# Schwefel: 10000, 10000, -2500, True, True, 9, 500, 0.01, 1e-6, 50
numSamples = 10000
numTrainingSamples = 10000
visualize = True
validation = False
nDimensions = 9
maxRange = 500
f = schwefel

C = 0.01
gamma = 1e-6
class_weight = {1: 10}

for threshold in range(100, 10, -10):
    print("Threshold: 100")

    clf = SVC(C=C, gamma=gamma, class_weight=class_weight)

    start = time.time()
    best, point, goodOptChance, numRuns, samples = lego(f, threshold=threshold, clf=clf, numSamples=numSamples,
                                                        numTrainingSamples=numTrainingSamples, n_dimensions=nDimensions,
                                                        maxRange=maxRange, visualize=visualize, validation=validation)

    json_data = {}

    json_data["lego"] = {}
    json_data["multistart"] = {}
    idx = 0
    value = 0
    print("Minimizing LeGO points.")
    for sample in samples:
        res = minimize(f, sample[0], method='L-BFGS-B', options={'ftol': 1e-8})
        json_data["lego"][str(idx)] = []
        json_data["lego"][str(idx)].append({"point": list(sample[0]),
                                            "value": res["fun"]})
        idx += 1
    print("Starting Multistart iterations.")
    for i in range(len(samples)):
        sys.stdout.write('\r Progress: {n}/{t}'.format(n=i, t=numSamples))
        sys.stdout.flush()
        value, point = multistartNew(f, nDimensions, maxRange, 1)
        json_data["multistart"][str(i)] = []
        json_data["multistart"][str(i)].append({"point": list(point),
                                                "value": value})
    print("\nDumping data to json.")
    with open("ECDF-R/rastrigin-{t}.json".format(t=threshold), "w+") as f:
        json.dump(json_data, f)