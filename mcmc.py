import copy
import random
import numpy as np
from bitstring import Bits, BitArray


class State(object):

    def __init__(self, length: int, value=None):
        self._length = length

        if value is None:
            self._value = self.generate_perm()
        else:
            self._value = value

        self._bitstring = BitArray(Bits(uint=self._value, length=self._length))

    def __len__(self):
        return self._length

    def __str__(self):
        return self.get_bitstring()

    def get_length(self):
        return self._length

    def get_bitstring(self):
        return self._bitstring.bin

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def generate_perm(self) -> int:
        """Takes a _length as input, and returns a randomly generated _number with binary _number _length = _length"""
        permutation = random.randint(0, (2 ** self._length)-1)
        return permutation

    def flip(self):
        flip_index = random.randint(0, self._length - 1)
        self._bitstring.invert(flip_index)


class Metropolis(object):

    def __init__(self, walkers, x: State):
        self.walkers = walkers
        self.x_new = copy.deepcopy(x)
        self.x_old = copy.deepcopy(x)
        self._state_length = self.x_new.get_length()

    def metropolis(self):
        # TODO sjekke forskjell mellom reset av bit_string og ikke, mellom hver kjøring

        for i in range(self._state_length):
            self.x_new.flip()

            if self.acceptance_criterion(self.x_new):
                self.x_old.set_value(self.x_new.get_value())
            else:
                self.x_new.set_value(self.x_old.get_value())

        return self.x_new

    def acceptance_criterion(self, function, sigma, mu) -> bool:
        u = random.uniform(0, 1)

        new_score = function(self.x_new.get_value(), sigma, mu)
        old_score = function(self.x_old.get_value(), sigma, mu)

        print('u: ' + str(u))
        print('old_score: ' + str(old_score))
        print('New_score: ' + str(new_score))

        score = new_score / old_score > u

        print(score)
        print('\n\n')

        return score


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


# generic function takes op and its argument



def normal_distribution(x: int, sigma: float, mu: float) -> float:
        """

        :param x:
        :param sigma:
        :param mu:
        """
        _1 = 1 / (sigma * np.sqrt(2 * np.pi))
        _2 = -(1 / 2) * ((x - mu) / sigma) ** 2

        return _1 * np.exp(_2)




