import json

from matplotlib import pyplot
from statsmodels.distributions.empirical_distribution import ECDF


def printECDFStats(function='rastrigin'):

    with open("{p}.json".format(p=function), "r") as f:
        json_data = json.load(f)
        f.close()

    threshold = json_data["stats"]["threshold"]
    numAcceptablesLego = 0
    numAcceptablesMS = 0
    numSamples = json_data["stats"]["numSamples"]

    bestLegoOptimum = float('inf')
    bestMultistartOptimum = float('inf')
    optimaLego = []
    optimaMultistart = []

    for point in json_data["lego"]:
        if json_data["lego"][str(point)][0]["value"] <= threshold:
            numAcceptablesLego += 1
        if json_data["lego"][str(point)][0]["value"] < bestLegoOptimum:
            bestLegoOptimum = json_data["lego"][str(point)][0]["value"]
        optimaLego.append(json_data["lego"][str(point)][0]["value"])

    for point in json_data["multistart"]:
        if json_data["multistart"][str(point)][0]["value"] <= threshold:
            numAcceptablesMS += 1
        if json_data["multistart"][str(point)][0]["value"] < bestMultistartOptimum:
            bestMultistartOptimum = json_data["multistart"][str(point)][0]["value"]
        optimaMultistart.append(json_data["multistart"][str(point)][0]["value"])

    print("----------------------------------")
    print("Problem Parameters:")
    print("Function: {p}".format(p=function))
    print("Dimensions: {p}".format(p=str(json_data["stats"]["nDimensions"] + 1)))
    print("Threshold: {p}".format(p=threshold))
    print("NumSamples: {p}".format(p=numSamples))
    print("MaxRange: {p}".format(p=json_data["stats"]["maxRange"]))
    print("----------------------------------")
    print("SVM Parameters for Lego:")
    print("C: {p}".format(p=json_data["stats"]["C"]))
    print("Gamma: {p}".format(p=json_data["stats"]["gamma"]))
    print("Positive examples class weight: {p}".format(p=json_data["stats"]["class_weight"]))
    print("Kernel: {p}".format(p=json_data["stats"]["kernel"]))
    print("----------------------------------")
    print("Percentage of acceptable minima using LeGO: {p}%".format(p=(numAcceptablesLego / numSamples)*100))
    print("Percentage of acceptable minima using Multistart: {p}%".format(p=(numAcceptablesMS / numSamples)*100))
    print("----------------------------------")
    print("Best Lego Optimum: {p}".format(p=bestLegoOptimum))
    print("Best Multistart Optimum: {p}".format(p=bestMultistartOptimum))

    # Calculating Empirical Cumulative Distribution Function for the problem optima

    ecdfLego = ECDF(optimaLego)
    ecdfMultistart = ECDF(optimaMultistart)

    pyplot.plot(ecdfLego.x, ecdfLego.y)
    pyplot.plot(ecdfMultistart.x, ecdfMultistart.y)
    pyplot.legend(['Lego', 'Multistart'])
    pyplot.title('ECDF Lego vs Multistart with {p} function'.format(p=function))
    pyplot.show()


if __name__ == '__main__':

    printECDFStats(function='rastrigin')

