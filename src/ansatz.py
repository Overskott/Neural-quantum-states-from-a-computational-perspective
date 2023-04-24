import numpy as np

import src.utils as utils
from config_parser import get_config_file


class RBM(object):

    def __init__(self,
                 visible_size=None,
                 hidden_size=None,
                 visible_bias=None,
                 hidden_bias=None,
                 hamiltonian=None,
                 weights: np.ndarray = None):

        data = get_config_file()['parameters']  # Load the config file
        scale = 1#/np.sqrt(n_hid)
        self.hamiltonian = hamiltonian

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
            self.b_r = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias #
            self.b_i = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias #
        else:
            self.b_r = np.real(visible_bias)
            self.b_i = np.imag(visible_bias)

        if hidden_bias is None:
            self.c_r = np.random.normal(0, 1/scale, (1, self.visible_size))  # Hidden layer bias
            self.c_i = np.random.normal(0, 1/scale, (1, self.visible_size))  # Hidden layer bias
        else:
            self.c_r = np.real(hidden_bias)
            self.c_i = np.imag(hidden_bias)

        if weights is None:
            self.W_r = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights
            self.W_i = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights
        else:
            self.W_r = np.real(weights)
            self.W_i = np.imag(weights)

        self.params = [self.b_r, self.b_i, self.c_r, self.c_i, self.W_r, self.W_i]

        #self.state = utils.random_binary_array(2**self.visible_size)
        # Generate all possible states
        all_states_list = []
        for i in range(2 ** self.visible_size):
            state = utils.numberToBase(i, 2, self.visible_size)
            all_states_list.append(state)
        self.all_states = np.array(all_states_list)

    def get_parameters_as_array(self):
        """Creates a variable array from the RBM variables"""
        real_part = np.concatenate((self.b_r, self.c_r, self.W_r.flatten()))
        imag_part = np.concatenate((self.b_i, self.c_i, self.W_i.flatten()))

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
            raise ValueError("Array myst be of correct n.")

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


    def probability(self, state: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """
        return np.abs(self.normalized_amplitude(state)) ** 2

    #@profile
    def amplitude(self, state: np.ndarray) -> np.ndarray:
        """ Calculates the amplitude of finding the RBM in state s """

        product = 1
        b = self.b_r+1j*self.b_i
        c = (self.c_r+1j*self.c_i)
        W = self.W_r+1j*self.W_i

        for i in range(self.hidden_size):
            scalar = -(W[:, i] @ state + c[i])
            product *= (1 + np.exp(scalar))

        bias = np.exp(np.transpose(b) @ state)

        amp = product * bias

        return amp

    def unnormalized_amplitude(self, state):
        Wstate = np.matmul(state, self.W_r) + 1j * np.matmul(state, self.W_i)
        exponent = Wstate + self.c_r + 1j * self.c_i
        A = np.exp(-exponent)
        A = np.prod(1 + A, axis=1, keepdims=True)
        A = A * np.exp(-np.matmul(state, self.b_r) - 1j * np.matmul(state, self.b_i))
        return A

    def normalized_amplitude(self, state):
        # Normalized amplitude
        Z = np.sqrt(np.sum(np.abs(self.unnormalized_amplitude(self.all_states)) ** 2))
        return self.unnormalized_amplitude(state) / Z

    def wave_function(self):
        return self.amplitude(self.all_states)

    def local_energy(self, state):
        batch_size = state.shape[0]
        E = np.zeros((batch_size, 1), dtype=np.complex128)
        a1 = self.amplitude(state)

        powers = np.array([2 ** i for i in reversed(range(self.n_vis))]).reshape(1, -1)
        state_indicies = np.sum(state * powers, axis=1)
        for i in range(2 ** self.n_vis):
            state_prime = np.array(numberToBase(i, 2, self.n_vis)).reshape(1, -1)
            a2 = self.amplitude(state_prime)

            h_slice = (self.hamiltonian[state_indicies, i]).reshape(-1, 1)
            E += (h_slice / a1) * a2

        return E

    def exact_energy(self):
        wave_function = self.wave_function()
        E = wave_function.conj().T @ self.hamiltonian @ wave_function
        return E.real

    def probability_fast(self, dist: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """
        return np.abs(self.amplitude_fast(dist)) ** 2

    #@profile
    def amplitude_fast(self, distribution: np.ndarray) -> np.ndarray:
        """ Calculates the amplitude of finding the RBM in state s """

        D = distribution

        product = 1
        b = self.b_r + 1j * self.b_i
        c = (self.c_r + 1j * self.c_i).reshape(-1, 1)
        W = self.W_r + 1j * self.W_i

        M = -(W @ D.T + c)
        np.prod(1 + np.exp(M), axis=0)
        bias = np.exp(np.transpose(b) @ D.T)

        amp = product * bias

        return amp

