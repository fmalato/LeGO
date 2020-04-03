import json

with open("schwefel.json", "r") as f:
    json_data = json.load(f)
    f.close()

threshold = -2500
numAcceptablesLego = 0
numAcceptablesMS = 0
numSamples = 10000
for point in json_data["lego"]:
    if json_data["lego"][str(point)][0]["value"] <= threshold:
        numAcceptablesLego += 1

for point in json_data["multistart"]:
    if json_data["multistart"][str(point)][0]["value"] <= threshold:
        numAcceptablesMS += 1

print("Percentage of acceptable minima using LeGO: {p}%".format(p=(numAcceptablesLego / numSamples)*100))
print("Percentage of acceptable minima using Multistart: {p}%".format(p=(numAcceptablesMS / numSamples)*100))
