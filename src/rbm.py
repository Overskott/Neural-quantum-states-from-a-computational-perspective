import numpy as np

import src.utils as utils
from src.state import State
from src.mcmc import Walker
from config_parser import get_config_file


class RBM(object):

    def __init__(self, visible_bias=None, hidden_bias=None, weights: np.ndarray = None):

        data = get_config_file()['parameters']  # Load the config file

        self.visible_size = data['visible_size']  # Get number of visible nodes from the config file
        self.state = utils.random_complex_array(self.visible_size)  # Set the initial state to a random complex array

        if visible_bias is None:
            self.b = np.random.uniform(-1, 1, self.visible_size)  # Visible layer bias #
        else:
            self.b = visible_bias

        if hidden_bias is None:
            self.c = np.random.uniform(-1, 1, self.visible_size)  # Hidden layer bias
        else:
            self.c = hidden_bias

        if weights is None:
            self.W = np.random.rand(self.visible_size, self.visible_size)  # s - h weights
        else:
            self.W = weights

    def set_visible(self, state):
        self.state = state

    def get_variable_array(self):
        """Creates a variable array from the RBM variables"""
        return np.concatenate((self.b, self.c, self.W.flatten()))  # Flattening the weights matrix

    def set_variables_from_array(self, x_0: np.ndarray):
        """
        Sets the RBM variables to the values in x_0

        b = x_0[:len(self.b)] is the visible layer bias
        c = x_0[len(self.b):len(self.b)+len(self.c)] is the hidden layer bias
        W = x_0[:len(self.b)+len(self.c)] is the weights
        """

        dim_0, dim_1 = np.shape(self.W)  # dim_0 visible layer, dim_1 hidden layer

        if len(x_0) != dim_0 * dim_1 + dim_0 + dim_1:
            raise ValueError("Array myst be of correct size.")

        self.b = x_0[:len(self.b)]
        self.c = x_0[len(self.b):len(self.b) + len(self.c)]
        self.W = x_0[len(self.b)+len(self.c):].reshape(dim_0, dim_1)

    def set_visible_bias(self, b):
        self.b = b

    def set_hidden_bias(self, c):
        self.c = c

    def set_weights(self, W):
        self.W = W

    def probability(self, configuration: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """

        return np.abs(self.amplitude(configuration)) ** 2

    def amplitude(self, state: np.ndarray) -> float:
        """ Calculates the amplitude of finding the RBM in state s """
        product = 1

        for i in range(self.visible_size):
            scalar = (self.W[:, i] @ state) + self.c[i]
            product *= (1 + np.exp(-scalar))

        bias = np.exp(np.transpose(self.b) @ state)

        amp = product * bias

        return amp

    # def probability(self, configuration: np.ndarray) -> float:
    #     """ Calculates the probability of finding the RBM in state s """
    #     product = 1
    #
    #     for i in range(self.visible_size):
    #         scalar = (self.W[:, i] @ configuration) + self.c[i]
    #         product *= (1 + np.exp(-scalar - self.c[i]))
    #
    #     bias = np.exp(np.transpose(self.b) @ configuration)
    #
    #     return product * bias

    def local_energy(self, hamiltonian, spin_config: np.ndarray) -> float:
        """Calculates the local energy of the RBM in state s"""

        h_size = hamiltonian.shape[0]
        i = spin_config
        local_state = spin_config
        local_energy = 0

        for j in range(h_size):
            p_i = self.probability(local_state)
            p_j = self.probability(utils.int_to_binary_array(j, spin_config.size))

            h_ij = hamiltonian[i, j]

            local_energy += h_ij * p_j/p_i

        return local_energy

    def get_rbm_energy(self, walker: Walker, hamiltonian):
        distribution = walker.get_history()
        energy = 0
        for state in distribution:
            energy += self.local_energy(hamiltonian, state)

        return energy / len(distribution)

    def minimize_energy(self, x_0: np.ndarray, *args):

        self.set_variables_from_array(x_0)
        walker = args[0]
        hamiltonian = args[1]

        walker.clear_history()
        walker.random_walk(self.probability)

        return self.get_rbm_energy(walker, hamiltonian)





