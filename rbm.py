import random
import numpy as np


class RBM(object):

    def __init__(self, visible_layer, visible_bias=None, hidden_layer=None, hidden_bias=None, weights=None):

        self.s = visible_layer
        self.n = len(self.s)
        if visible_bias is None:
            self.b = np.random.rand(self.n)  # Visible layer bias
        else:
            self.b = visible_bias

        if hidden_layer is None:
            self.h = np.asarray([random.randint(0, 1) for _ in range(self.n)])  # Hidden layer state
        else:
            self.h = hidden_layer

        if hidden_bias is None:
            self.c = np.random.rand(self.n)  # Hidden layer bias
        else:
            self.c = hidden_bias

        if weights is None:
            self.W = np.random.rand(self.n, self.n)  # s - h weights
        self.z = self.n  # normalization TODO Ask about this!

    def energy(self, state):
        '''Calculates the RBM energy'''

        e = 0
        e += np.transpose(self.h) @ self.W @ state
        e += np.transpose(self.c) @ self.h  # TODO Here we need the hidden layers to calculate energy?
        e += np.transpose(self.b) @ state
        return e

    def probability(self, state):
        ''' Calculates the probability of finding the RBM in state s '''  #  TODO is it the probability of findig system is state s?
        product = 1
        for i in range(self.n):
            scalar = (self.W[i, :] @ np.transpose(state))
            product *= (1 + np.exp(-scalar - self.c[i]))

        bias = np.exp(np.transpose(self.b) @ state)

        return 1/self.z * product * bias
