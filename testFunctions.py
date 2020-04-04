import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def rosen(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


def rastrigin(x, A=10):
    res = A * len(x)
    for i in range(len(x)):
        res += x[i] ** 2 - A * np.cos(2 * math.pi * x[i])
    return res


'''
def schwefel(x):
    if all(value < 500 for value in x) and all(value > 500 for value in x):
        res = 418.9829 * len(x)
        for i in range(len(x)):
            res -= x[i] * np.sin(np.sqrt(np.absolute(x[i])))
        return res
    else:
        return float('inf')
'''


def schwefel(x):
    try:
        assert np.all(np.abs(x) <= 500)
        return np.sum(np.prod([x, -np.sin(np.sqrt(np.abs(x)))], axis=0), axis=-1)
    except:
        return float('inf')


def demoOptimization(f=rosen, x0=np.array([0.1, 0.1]), visualize=True, name='rosen'):
    res = minimize(f, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

    if visualize:
        if name == 'rosen':
            x = np.arange(-2, 2, 0.1)
            y = np.arange(-1, 3, 0.1)
        elif name == 'rastrigin':
            x = np.arange(-5.12, 5.12, 0.1)
            y = np.arange(-5.12, 5.12, 0.1)
        else:
            x = np.arange(-500, 500, 10)
            y = np.arange(-500, 500, 10)

        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, -45)
        results = f(xy)
        ax.plot_surface(xgrid, ygrid, results, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        ax2.plot(x, results)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot()
        ax3.set_xlabel('y')
        ax3.set_ylabel('z')
        ax3.plot(y, results)

        ax.scatter(res.x[0], res.x[1], res['fun'], marker='x')
        plt.show()


if __name__ == '__main__':
    x0 = np.array([3, 2])
    demoOptimization(f=schwefel, x0=x0, name='schwefel')
