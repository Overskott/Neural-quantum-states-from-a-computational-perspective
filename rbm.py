import random
import numpy as np

import utils
from state import State
from mcmc import Walker


class RBM(object):

    def __init__(self, visible_layer, visible_bias=None, hidden_bias=None, weights=None):

        self.s = visible_layer
        self.n = len(self.s)
        if visible_bias is None:
            self.b = np.random.uniform(-1, 1, self.n)  # Visible layer bias #
        else:
            self.b = visible_bias

        if hidden_bias is None:
            self.c = np.random.uniform(-1, 1, self.n)  # Hidden layer bias
        else:
            self.c = hidden_bias

        if weights is None:
            self.W = np.random.rand(self.n, self.n)  # s - h weights
        else:
            self.W = weights

    def set_visible(self, state):
        self.s = state

    def probability(self, bit_array: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """
        product = 1

        for i in range(self.n):
            scalar = (self.W[i, :] @ bit_array) + self.c[i]
            product *= (1 + np.exp(-scalar - self.c[i]))

        bias = np.exp(np.transpose(self.b) @ bit_array)

        return product * bias

    def local_energy(self, hamiltonian, spin_config: State):
        """Calculates the local energy of the RBM in state s"""

        h_size = hamiltonian.shape[0]
        i = spin_config.get_value()
        local_state = spin_config.get_bit_array()
        local_energy = 0

        for j in range(h_size):
            p_i = self.probability(local_state)
            p_j = self.probability(utils.int_to_binary_array(j, spin_config.get_length()))

            h_ij = hamiltonian[i, j]

            local_energy += float(h_ij * np.sqrt(p_j/p_i))

        return local_energy

    def get_rbm_energy(self, walker: Walker, hamiltonian):

        distribution = walker.get_walk_results()
        energy = 0
        for state in distribution:
            energy += self.local_energy(hamiltonian, state)

        return energy / len(distribution)

