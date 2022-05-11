# This is a sample Python script.
import random
import numpy as np
from bit_string import BitString
import matplotlib.pyplot as plt

def normal_distribution(x: int, sigma: float, mu: float) -> float:
    """

    :param x:
    :param sigma:
    :param mu:
    """
    _1 = 1/(sigma*np.sqrt(2*np.pi))
    _2 = -(1/2)*((x-mu)/sigma)**2
    return _1*np.exp(_2)


class MCMC:

    def __init__(self, n, bit_length):
        self.n = n
        self.bit_string = BitString(bit_length)
        self.distribution = normal_distribution

    def find_x_new(self):
        self.bit_string.flip()

    def acceptance_criterion(self, x_new) -> bool:
        u = random.uniform(0, 1)

        mu = self.bit_string.get_length()/2

        return self.distribution(x_new, 1, mu) / self.distribution(self.bit_string, 1, mu) > u

    def metropolis(self):
        # TODO sjekke forskjell mellom reset av bit_string og ikke, mellom hver kjøring
        x_new = self.bit_string

        for i in range(self.n):
            self.find_x_new()

            if self.acceptance_criterion(x_new):
                self.bit_string = x_new

        return x_new

    def average(self):
        x_hat = 0
        for i in range(self.n):
            x_hat = x_hat + self.metropolis()
        return x_hat/self.n


if __name__ == '__main__':

    N = 100
    length = 7

    mcmc = MCMC(N, length)
    mcmc.metropolis()
