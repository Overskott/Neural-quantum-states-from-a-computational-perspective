import numpy as np

import src.utils as utils
from config_parser import get_config_file


class RBM(object):

    def __init__(self,
                 visible_size=None,
                 hidden_size=None,
                 visible_bias=None,
                 hidden_bias=None,
                 weights: np.ndarray = None):

        data = get_config_file()['parameters']  # Load the config file

        if visible_bias is not None:
            self.visible_size = len(visible_bias)
        elif visible_size is None:
            self.visible_size = data['visible_size']
        else:
            self.visible_size = visible_size

        if hidden_bias is not None:
            self.hidden_size = len(hidden_bias)
        elif hidden_size is None:
            self.hidden_size = data['hidden_size']
        else:
            self.hidden_size = hidden_size

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


        self.state = utils.random_binary_array(self.visible_size)

    def set_visible(self, state):
        self.state = state

    def get_parameters_as_array(self):
        """Creates a variable array from the RBM variables"""
        return np.concatenate((self.b, self.c, self.W.flatten()))  # Flattening the weights matrix

    def set_parameters_from_array(self, x_0: np.ndarray):
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

    def set_parameter_from_value(self, index, value):
        """ Sets the parameter at index to value """

        v_size = self.visible_size
        h_size = self.hidden_size

        if index < v_size:
            self.b[index] = value

        elif index < v_size + h_size:
            self.c[index - v_size] = value

        else:
            w_index = index - v_size - h_size

            row = w_index // h_size
            column = w_index % h_size

            self.W[row, column] = value

    def set_visible_bias(self, b):
        self.b = b

    def set_hidden_bias(self, c):
        self.c = c

    def set_weights(self, W):
        self.W = W

    def probability(self, state: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """

        return np.abs(self.amplitude(state)) ** 2

    def amplitude(self, state: np.ndarray) -> float:
        """ Calculates the amplitude of finding the RBM in state s """
        product = 1

        for i in range(self.visible_size):
            scalar = (self.W[:, i] @ state) + self.c[i]
            product *= (1 + np.exp(-scalar))

        bias = np.exp(np.transpose(self.b) @ state)

        amp = product * bias

        return amp







