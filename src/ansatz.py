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
            self.b_r = np.random.uniform(-1, 1, self.visible_size)  # Visible layer bias #
            self.b_i = np.random.uniform(-1, 1, self.visible_size)  # Visible layer bias #
        else:
            self.b_r = np.real(visible_bias)
            self.b_i = np.imag(visible_bias)

        if hidden_bias is None:
            self.c_r = np.random.uniform(-1, 1, self.visible_size)  # Hidden layer bias
            self.c_i = np.random.uniform(-1, 1, self.visible_size)  # Hidden layer bias
        else:
            self.c_r = np.real(hidden_bias)
            self.c_i = np.imag(hidden_bias)

        if weights is None:
            self.W_r = np.random.rand(self.visible_size, self.hidden_size)  # s - h weights
            self.W_i = np.random.rand(self.visible_size, self.hidden_size)  # s - h weights
        else:
            self.W_r = np.real(weights)
            self.W_i = np.imag(weights)

        self.state = utils.random_binary_array(2**self.visible_size)

    def set_visible(self, state):
        self.state = state

    def get_parameters_as_array(self):
        """Creates a variable array from the RBM variables"""
        real_part = np.concatenate((self.b_r, self.c_r, self.W_r.flatten()))  # Flattening the weights matrix
        imag_part = np.concatenate((self.b_i, self.c_i, self.W_i.flatten()))  # Flattening the weights matrix

        return np.concatenate((real_part, imag_part))

    def set_parameters_from_array(self, x: np.ndarray):
        """
        Sets the RBM variables to the values in x_0

        b = x_0[:len(self.b)] is the visible layer bias
        c = x_0[len(self.b):len(self.b)+len(self.c)] is the hidden layer bias
        W = x_0[:len(self.b)+len(self.c)] is the weights
        """

        x_r = x[:len(x) // 2]
        x_i = x[len(x) // 2:]

        dim_0, dim_1 = np.shape(self.W_r)  # dim_0 visible layer, dim_1 hidden layer

        if len(x_r) != dim_0 * dim_1 + dim_0 + dim_1:
            raise ValueError("Array myst be of correct size.")

        self.b_r = x_r[:self.visible_size]
        self.c_r = x_r[self.visible_size:self.visible_size + self.hidden_size]
        self.W_r = x_r[self.visible_size+self.hidden_size:].reshape(dim_0, dim_1)

        self.b_i = x_i[:self.visible_size]
        self.c_i = x_i[self.visible_size:self.visible_size + self.hidden_size]
        self.W_i = x_i[self.visible_size + self.hidden_size:].reshape(dim_0, dim_1)

    def set_parameter_from_value(self, index, value):
        """ Sets the parameter at index to value """

        v_size = self.visible_size
        h_size = self.hidden_size

        real_size = len(self.get_parameters_as_array())//2

        if index < real_size:
            if index < v_size:
                self.b_r[index] = value

            elif index < v_size + h_size:
                self.c_r[index - v_size] = value

            else:
                w_index = index - v_size - h_size

                row = w_index // h_size
                column = w_index % h_size

                self.W_r[row, column] = value
        else:
            imag_index = index - real_size

            if imag_index < v_size:
                self.b_i[imag_index] = value

            elif index < v_size + h_size:
                self.c_i[imag_index - v_size] = value

            else:
                w_index = imag_index - v_size - h_size

                row = w_index // h_size
                column = w_index % h_size

                self.W_i[row, column] = value
    #
    # def set_visible_bias(self, b):
    #     self.b = b
    #
    # def set_hidden_bias(self, c):
    #     self.c = c
    #
    # def set_weights(self, W):
    #     self.W = W

    def probability(self, state: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """

        return np.abs(self.amplitude(state)) ** 2

    def amplitude(self, state: np.ndarray) -> float:
        """ Calculates the amplitude of finding the RBM in state s """
        product = 1
        b = self.b_r+1j*self.b_i
        c = self.c_r+1j*self.c_i
        W = self.W_r+1j*self.W_i

        for i in range(self.hidden_size):

            scalar = -(W[:, i] @ state + c[i])
            product *= (1 + np.exp(scalar))

        bias = np.exp(np.transpose(b) @ state)

        amp = product * bias

        return amp







