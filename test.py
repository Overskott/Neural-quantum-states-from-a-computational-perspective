import numpy as np
import matplotlib.pyplot as plt
from rbm import RBM
from state import State

from utils import *


def normal_distribution(x, sigma, mu) -> float:
    _1 = 1 / (sigma * np.sqrt(2 * np.pi))
    _2 = -(1 / 2) * ((x - mu) / sigma) ** 2

    return _1 * np.exp(_2)

L = 10
sigma = 1
mu = 0
x = np.linspace(-L, L, 100)
y = normal_distribution(x, sigma, mu)

discrete = np.linspace(-L, L, 30)

gauss_array = np.logspace(-L, .6, 30)
print(gauss_array)


gauss = np.concatenate((gauss_array, -gauss_array))

plt.plot(x, y)
plt.scatter(discrete, normal_distribution(discrete, sigma, mu), color='red')
#plt.scatter(gauss, normal_distribution(gauss, sigma, mu), color='green')
plt.show()
