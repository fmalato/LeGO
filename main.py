# Temporarily using this file to run some tests before the final release

import random as rand
import numpy as np
import pandas as pd

space_dimension = 3

def f(x):
    # objective function definition (x^2 + 3z, for now)
    return pow(x[0], 2) + 3 * x[2]

def generate():
    # for now it generates a bunch of random numbers
    return [rand.randint(1, 10) for x in range(space_dimension)]


gamma = 0.01  # Step size multiplier
precision = 0.00001  # Desired precision of result
max_iters = 10000  # Maximum number of iterations

def df(x):
    # derivative wrt x[0], just to test if refine() works
    return 2 * x[0] + 3 * x[2]

def refine(x):
    next_x = x  # We start the search at x=6
    current_x = [0 for n in range(space_dimension)]
    for _ in range(max_iters):
        for i in range(space_dimension):
            current_x[i] = next_x[i]
            next_x[i] = current_x[i] - gamma * df(current_x)

            step = next_x[i] - current_x[i]
            if abs(step) <= precision:
                break
    return current_x

def algorithm1():
    f_star = float('inf')
    k = 0
    x_star = 0
    while k < 3:
        x = generate()
        y = refine(x)
        if f(y) < f_star:
            f_star = f(y)
            x_star = y[k]
        k += 1
        print(x, y)
    return x_star, f_star

x_star = f_star = 10000000
iter_x  = 0
iter_f = 0
for i in range(30):
    x, fa = algorithm1()
    if x < x_star:
        x_star = x
        iter_x = i
    if fa < f_star:
        f_star = fa
        iter_f = i


print((x_star, str(iter_x) + '/30'), (f_star, str(iter_f) + '/30'))
