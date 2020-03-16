import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def rastrigin(x, A=10):
    return A + sum(x[:-1]**2 - A * np.cos(2 * math.pi * x[:-1]))


def demoOptimization(f=rosen, x0=np.array([0.1, 0.1, 0.1]), visualize=True):
    res = minimize(f, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    print(res)

    if visualize:
        x = np.arange(-5.12, 5.12, 0.1)
        y = np.arange(-5.12, 5.12, 0.1)
        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45, -45)
        ax.plot_surface(xgrid, ygrid, f(xy), cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')

        print(res.x[0])

        ax.scatter(res.x[0], res.x[1], f(np.stack([res.x[0], res.x[1]])), marker='x')
        plt.show()


x0 = np.array([2.1, 2.1, 3.1])
demoOptimization(f=rastrigin, x0=x0)


'''

Old

def rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return A * len(X) + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])


def der_rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return sum([(2 * x + A * 2 * math.pi * np.sin(2 * math.pi * x)) for x in X])
    

def demoPlot():
    X = np.linspace(-5.12, 5.12)
    Y = np.linspace(-5.12, 5.12)

    X, Y = np.meshgrid(X, Y)

    Z = rastrigin(X, Y)
    dZ = der_rastrigin(X, Y, A=10)

    fig = plt.figure()
    fig2 = plt.figure()

    ax = fig.gca(projection='3d')
    ax2 = fig2.gca(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
    ax2.plot_surface(X, Y, dZ, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)

    plt.show()

'''