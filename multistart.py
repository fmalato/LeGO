import numpy as np
import time

from scipy.optimize import minimize
from numpy import random
from testFunctions import rosen, rastrigin, schwefel

"""
    IDEA:
    1. sample a point X in S (uniformly)
    2. start a local optimization from X
    3. accept the obtained local optimum Y if it is an improvement
    4. stop after N local searches
"""


def multistart(f, x, numSamples=100):
    indices = range(len(x))
    # I decided to go with the indices since the dimension of the vector may change
    actualBest = float('inf')
    bestPoint = []
    for i in range(numSamples):
        # selecting the actual sample
        sample = x[random.randint(low=indices[0], high=indices[len(indices) - 1])]
        # local search (cannot find Newton?!)
        res = minimize(f, sample, method='nelder-mead', options={'xatol': 1e-8})
        # getting the best local optimum and its value...
        if res['fun'] < actualBest:
            actualBest = res['fun']
            bestPoint = sample
    # ... and returning them
    return actualBest, bestPoint


def generate(n_dimensions, maxRange):
    rangify = np.vectorize(lambda v: v * (2 * maxRange) - maxRange)
    return rangify(np.random.rand(n_dimensions))


def multistartNew(f, n_dimensions=2, maxRange=5.12, numSamples=100):
    actualBest = float('inf')
    bestPoint = []
    for i in range(numSamples):
        sample = generate(n_dimensions, maxRange)
        res = minimize(f, sample, method='nelder-mead', options={'xatol': 1e-8})
        if res['fun'] < actualBest:
            actualBest = res['fun']
            bestPoint = res.x
    return actualBest, bestPoint


'''
x = np.arange(-5.12, 5.12, 0.1)
y = np.arange(-5.12, 5.12, 0.1)
xgrid, ygrid = np.meshgrid(x, y)
xy = np.stack([xgrid, ygrid])
z = rastrigin(xy)
data = []
for i in range(len(x)):
    for j in range(len(y)):
        data.append([xgrid[i][j], ygrid[i][j], z[i][j]])

data = np.asarray(data)
print(data.shape)
best, point = multistart(rastrigin, data, numSamples=100)
print('best: ' + str(best) + '    point: ' + str(point))

start = time.time()
best, point = multistartNew(rastrigin, n_dimensions=4, maxRange=5.12, numSamples=1000)
end = time.time()
print('best: ' + str(best) + '    point: ' + str(point) + '    time elapsed: ' + str(end - start))

'''
