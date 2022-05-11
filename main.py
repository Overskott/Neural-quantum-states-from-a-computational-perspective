# This is a sample Python script.
import copy
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
    _3 = _1*np.exp(_2)

    _4 = 1 / (sigma/2 * np.sqrt(2 * np.pi))
    _5 = -(1 / 2) * ((x - mu*2) / sigma/2) ** 2
    _6 = _4 * np.exp(_5)
    return _3+_6


class MCMC:

    def __init__(self, n, bit_length):
        self.n = n
        self.bit_length = bit_length
        self.bit_string = BitString(bit_length)
        self.distribution = normal_distribution

    def acceptance_criterion(self, x_new) -> bool:
        u = random.uniform(0, 1)

        sigma = self.bit_length
        mu = (2**self.bit_length)/5

        new_score = self.distribution(x_new.get_number(), sigma, mu)
        old_score = self.distribution(self.bit_string.get_number(), sigma, mu)

        print('u: ' + str(u))
        print('old_score: ' + str(old_score))
        print('New_score: ' + str(new_score))

        score = new_score / old_score > u

        print(score)
        print('\n\n')

        return score

    def metropolis(self):
        # TODO sjekke forskjell mellom reset av bit_string og ikke, mellom hver kjøring
        x_new = BitString(self.bit_length)
        # x_new.number = self.bit_string.number

        for i in range(self.n):
            x_new.flip()
            print('New number: ' + str(x_new.number))

            if self.acceptance_criterion(x_new):
                self.bit_string.number = x_new.number
            else:
                x_new.number = self.bit_string.number

        return x_new

    def average(self):
        x_hat = 0

        sigma = self.bit_length
        mu = (2 ** self.bit_length) / 5
        for i in range(self.n):
            run = self.metropolis()
            x_hat = x_hat + run.number
            # plt.bar(run.get_number(), normal_distribution(run, sigma, mu))

            plt.plot(i, normal_distribution(i, sigma, mu), 'r.')
            plt.plot(run.get_number(), normal_distribution(run.get_number(), sigma, mu), 'k.')

        plt.show()
        return x_hat/self.n


if __name__ == '__main__':

    N = 50
    length = 8

    mcmc = MCMC(N, length)

    print('average: ' + str(mcmc.average()))
