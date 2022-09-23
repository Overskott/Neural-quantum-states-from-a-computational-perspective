import copy
import random
import numpy as np
from state import State
from bitstring import Bits, BitArray




class Metropolis(object):

    def __init__(self, walker_steps, x: State, distribution):
        self.walker_steps = walker_steps
        self._state_length = x.get_length()
        self.x_new = copy.deepcopy(x)
        self.x_old = State(x.get_length())
        self.distribution = distribution

    def metropolis(self):
        # TODO sjekke forskjell mellom reset av bit_string og ikke, mellom hver kjøring
        accepted = 0
        for i in range(self.walker_steps):
            self.x_new.flip()
            if self.acceptance_criterion(self.distribution):
                self.x_old.set_value(self.x_new.get_value())
                accepted += 1
            else:
                self.x_new.set_value(self.x_old.get_value())
        accept_rate = accepted/self.walker_steps
        return self.x_new, accept_rate

    def acceptance_criterion(self, function) -> bool:
        u = random.uniform(0, 1)

        new_score = self.runOp(function, self.x_new.get_value())
        old_score = self.runOp(function, self.x_old.get_value())

        score = new_score / old_score > u

        return score

    def runOp(self, op, val):
        return op(val)


class MCMC(object):

    def __init__(self, n, bit_length):
        pass

    def average(self):
        x_hat = 0

        sigma = self.bit_length
        mu = (2 ** self.bit_length) / 5
        for i in range(self.n):
            run = self.metropolis()
            x_hat = x_hat + run._number
            # plt.bar(run.get_number(), normal_distribution(run, sigma, mu))

            #plt.plot(i, normal_distribution(i, sigma, mu), 'r.')
            #plt.plot(run.get_number(), normal_distribution(run.get_number(), sigma, mu), 'k.')

        # plt.show()
        return x_hat/self.n


def normal_distribution(x: int, sigma: float, mu: float) -> float:
        """

        :param x:
        :param sigma:
        :param mu:
        """
        _1 = 1 / (sigma * np.sqrt(2 * np.pi))
        _2 = -(1 / 2) * ((x - mu) / sigma) ** 2

        return _1 * np.exp(_2)


def double_normal_distribution(x: int, distance: int, sigma_1: float, mu_1: float, sigma_2: float, mu_2: float):

    return (normal_distribution(x, sigma_1, mu_1) + normal_distribution(x+distance, sigma_2, mu_2))/2
