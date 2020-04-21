import sys
import json

from lego import lego
from sklearn.svm import SVC
from scipy.optimize import minimize
from testFunctions import rastrigin, schwefel
from multistart import multistartNew
from stats import printECDFStats


def saveJsonStats(f=schwefel, numSamples=10000, numTrainingSamples=10000, threshold=-2500, nDimensions=9, maxRange=500,
                  C=0.01, gamma=1e-6, class_weight={1: 10}, validation=False, score='recall'):

    clf = SVC(C=C, gamma=gamma, class_weight=class_weight)

    best, point, goodOptChance, numRuns, samples, C, gamma, class_weight, kernel = lego(f, threshold=threshold, clf=clf,
                                                                                        numSamples=numSamples,
                                                                                        numTrainingSamples=numTrainingSamples,
                                                                                        n_dimensions=nDimensions,
                                                                                        maxRange=maxRange,
                                                                                        validation=validation,
                                                                                        score=score)

    json_data = {"lego": {}, "multistart": {}, "stats": {}}

    # Problem Stats
    json_data["stats"]["nDimensions"] = nDimensions
    json_data["stats"]["threshold"] = threshold
    json_data["stats"]["numSamples"] = numSamples
    json_data["stats"]["maxRange"] = maxRange

    # SVM stats
    json_data["stats"]["C"] = C
    json_data["stats"]["gamma"] = gamma
    json_data["stats"]["class_weight"] = class_weight
    json_data["stats"]["kernel"] = kernel

    idx = 0
    print("Minimizing LeGO points.")
    for sample in samples:
        res = minimize(f, sample[0], method='L-BFGS-B', options={'ftol': 1e-8})
        json_data["lego"][str(idx)] = []
        json_data["lego"][str(idx)].append({"point": list(sample[0]), "value": res["fun"]})
        idx += 1

    print("Starting Multistart iterations.")
    for i in range(len(samples)):
        sys.stdout.write('\r Progress: {n}/{t}'.format(n=i, t=numSamples))
        sys.stdout.flush()
        value, point = multistartNew(f, nDimensions, maxRange, 1)
        json_data["multistart"][str(i)] = []
        json_data["multistart"][str(i)].append({"point": list(point), "value": value})

    print("\nDumping data to json.")
    with open("{p}.json".format(p=f.__name__), "w+") as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    # Rastrigin: 10000, 10000, 40, True, True, 9, 5.12, 0.1, 0.01, 10
    # Schwefel: 10000, 10000, -2500, True, True, 9, 500, 0.01, 1e-6, 10

    numSamples = 10000
    numTrainingSamples = 10000
    threshold = -2500
    visualize = True
    validation = False
    nDimensions = 9
    maxRange = 500
    f = schwefel

    # La validation si puo' anche evitare se si conoscono gia' i parametri, basta inserirli nella funzione. La curva
    # viene meglio se si ottimizza la precision, anche se servira' piu' tempo all'algoritmo

    saveJsonStats(f=f, threshold=-2500, maxRange=500, validation=True, score='precision')
    printECDFStats(f.__name__)

