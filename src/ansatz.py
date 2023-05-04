import numpy as np
from tqdm import tqdm

import src.utils as utils
from config_parser import get_config_file
from src.mcmc import Walker
from src.optimization import Adam


class RBM(object):

    def __init__(self,
                 visible_size=None,
                 hidden_size=None,
                 hamiltonian=None
                 ):

        data = get_config_file()['parameters']  # Load the config file
        scale = 1#/np.sqrt(n_hid)

        if visible_size is None:
            self.visible_size = data['visible_size']
        else:
            self.visible_size = visible_size

        if hidden_size is None:
            self.hidden_size = data['hidden_size']
        else:
            self.hidden_size = hidden_size

        if hamiltonian is None:
            self.hamiltonian = utils.random_hamiltonian(2**self.visible_size)
        else:
            self.hamiltonian = hamiltonian

        self.adam = Adam()

        self.b_r = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias #
        self.b_i = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias #

        self.c_r = np.random.normal(0, 1/scale, (1, self.hidden_size))  # Hidden layer bias
        self.c_i = np.random.normal(0, 1/scale, (1, self.hidden_size))  # Hidden layer bias

        self.W_r = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights
        self.W_i = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights

        self.params = [self.b_r, self.b_i, self.c_r, self.c_i, self.W_r, self.W_i]

        #self.state = utils.random_binary_array(2**self.visible_size)

        # Generate all possible states
        all_states_list = []

        for i in range(2 ** self.visible_size):
            state = utils.numberToBase(i, 2, self.visible_size)
            all_states_list.append(state)

        self.all_states = np.array(all_states_list)

    # def get_parameters_as_array(self):
    #     """Creates a variable array from the RBM variables"""
    #     real_part = np.concatenate((self.b_r, self.c_r, self.W_r.flatten()))
    #     imag_part = np.concatenate((self.b_i, self.c_i, self.W_i.flatten()))
    #
    #     return np.concatenate((real_part, imag_part))
    #
    # def set_parameters_from_array(self, x: np.ndarray):
    #     """
    #     Sets the RBM variables to the values in x_0
    #
    #     b = x_0[:len(self.b)] is the visible layer bias
    #     c = x_0[len(self.b):len(self.b)+len(self.c)] is the hidden layer bias
    #     W = x_0[:len(self.b)+len(self.c)] is the weights
    #     """
    #
    #     x_r = x[:len(x) // 2]
    #     x_i = x[len(x) // 2:]
    #
    #     dim_0, dim_1 = np.shape(self.W_r)  # dim_0 visible layer, dim_1 hidden layer
    #
    #     if len(x_r) != dim_0 * dim_1 + dim_0 + dim_1:
    #         raise ValueError("Array myst be of correct n.")
    #
    #     self.b_r = x_r[:self.visible_size]
    #     self.c_r = x_r[self.visible_size:self.visible_size + self.hidden_size]
    #     self.W_r = x_r[self.visible_size+self.hidden_size:].reshape(dim_0, dim_1)
    #
    #     self.b_i = x_i[:self.visible_size]
    #     self.c_i = x_i[self.visible_size:self.visible_size + self.hidden_size]
    #     self.W_i = x_i[self.visible_size + self.hidden_size:].reshape(dim_0, dim_1)
    #
    # def set_parameter_from_value(self, index, value):
    #     """ Sets the parameter at index to value """
    #
    #     v_size = self.visible_size
    #     h_size = self.hidden_size
    #
    #     real_size = len(self.get_parameters_as_array())//2
    #
    #     if index < real_size:
    #         if index < v_size:
    #             self.b_r[index] = value
    #
    #         elif index < v_size + h_size:
    #             self.c_r[index - v_size] = value
    #
    #         else:
    #             w_index = index - v_size - h_size
    #
    #             row = w_index // h_size
    #             column = w_index % h_size
    #
    #             self.W_r[row, column] = value
    #     else:
    #         imag_index = index - real_size
    #
    #         if imag_index < v_size:
    #             self.b_i[imag_index] = value
    #
    #         elif index < v_size + h_size:
    #             self.c_i[imag_index - v_size] = value
    #
    #         else:
    #             w_index = imag_index - v_size - h_size
    #
    #             row = w_index // h_size
    #             column = w_index % h_size
    #
    #             self.W_i[row, column] = value
    #
    #
    #
    #
    # #@profile
    # def amplitude_old(self, state: np.ndarray) -> np.ndarray:
    #     """ Calculates the amplitude_old of finding the RBM in state s """
    #
    #     product = 1
    #     b = self.b_r+1j*self.b_i
    #     c = (self.c_r+1j*self.c_i)
    #     W = self.W_r+1j*self.W_i
    #
    #     for i in range(self.hidden_size):
    #         scalar = -(W[:, i] @ state + c[i])
    #         product *= (1 + np.exp(scalar))
    #
    #     bias = np.exp(np.transpose(b) @ state)
    #
    #     amp = product * bias
    #
    #     return amp
    #
    #
    # def probability_fast(self, dist: np.ndarray) -> float:
    #     """ Calculates the probability of finding the RBM in state s """
    #     return np.abs(self.amplitude_fast(dist)) ** 2
    #
    # #@profile
    # def amplitude_fast(self, distribution: np.ndarray) -> np.ndarray:
    #     """ Calculates the amplitude_old of finding the RBM in state s """
    #
    #     D = distribution
    #
    #     product = 1
    #     b = self.b_r + 1j * self.b_i
    #     c = (self.c_r + 1j * self.c_i).reshape(-1, 1)
    #     W = self.W_r + 1j * self.W_i
    #
    #     M = -(W @ D.T + c)
    #     np.prod(1 + np.exp(M), axis=0)
    #     bias = np.exp(np.transpose(b) @ D.T)
    #
    #     amp = product * bias
    #
    #     return amp

    # From Kristians code

    def probability(self, state: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """
        return np.abs(self.normalized_amplitude(state)) ** 2

    def unnormalized_probability(self, state):
        return np.abs(self.unnormalized_amplitude(state))**2

    def unnormalized_amplitude(self, state):
        Wstate = np.matmul(state, self.W_r) + 1j * np.matmul(state, self.W_i)
        exponent = Wstate + self.c_r + 1j * self.c_i
        A = np.exp(-exponent)
        A = np.prod(1 + A, axis=1, keepdims=True)
        A = A * np.exp(-np.matmul(state, self.b_r) - 1j * np.matmul(state, self.b_i))
        return A

    def normalized_amplitude(self, state):
        # Normalized amplitude_old
        Z = np.sqrt(np.sum(np.abs(self.unnormalized_amplitude(self.all_states)) ** 2))
        return self.unnormalized_amplitude(state) / Z

    def wave_function(self):
        return self.normalized_amplitude(self.all_states)

    def probability_dist(self):
        return np.abs(self.wave_function())**2

    def local_energy(self, state):
        batch_size = state.shape[0]
        E = np.zeros((batch_size, 1), dtype=np.complex128)
        a1 = self.unnormalized_amplitude(state)

        powers = np.array([2 ** i for i in reversed(range(self.visible_size))]).reshape(1, -1)
        state_indices = np.sum(state * powers, axis=1)
        for i in range(2 ** self.visible_size):
            state_prime = np.array(utils.numberToBase(i, 2, self.visible_size)).reshape(1, -1)
            a2 = self.unnormalized_amplitude(state_prime)

            h_slice = (self.hamiltonian[state_indices, i]).reshape(-1, 1)
            E += (h_slice / a1) * a2

        return E

    def exact_energy(self):
        wave_function = self.wave_function()
        E = wave_function.conj().T @ self.hamiltonian @ wave_function
        return E.real

    def estimate_energy(self):
        walker = Walker()
        sampled_states = walker(self.probability, 1000)
        return np.mean(self.local_energy(sampled_states)).real

    def omega_estiamte(self, states):
        omega_list = []

        b_grad = self.b_grad(states)
        c_grad = self.c_grad(states)
        W_grad = self.W_grad(states)

        omega_list.extend([b_grad, 1j * b_grad])

        omega_list.extend([c_grad, 1j * c_grad])

        omega_list.extend([W_grad, 1j * W_grad])

        return omega_list

    def omega(self, states):
        omega_list = []

        b_grad = self.b_grad(states).T
        c_grad = self.c_grad(states).T
        W_grad = self.W_grad(states).T

        A = self._diag(b_grad)
        omega_list.extend([A, 1j * A])

        A = self._diag(c_grad)
        omega_list.extend([A, 1j * A])

        A = self._diag(W_grad)
        omega_list.extend([A, 1j * A])

        return omega_list

    def _diag(self, A):
        # hack to make batch of vectors into batch of diagonal matrices
        num_params = A.shape[1]
        A = np.expand_dims(A, axis=1)
        A = A * np.eye(num_params)
        return A

    def b_grad(self, state):
        return -state

    def c_grad(self, state):
        exponent = np.matmul(state, self.W_r) + 1j * np.matmul(state, self.W_i)
        exponent += self.c_r + 1j * self.c_i
        A = -np.exp(-exponent) / (1 + np.exp(-exponent))
        return A

    def W_grad(self, state):
        batch_size = state.shape[0]
        A = self.c_grad(state)
        # batch-wise outer product between c_grad and state
        A = np.einsum('ij,ik->ijk', state, A).reshape(batch_size, -1)
        return A

    def finite_grad(self, h=0.05):
        grad_list = []
        for param in self.params:
            grad_array = np.zeros(param.shape)
            for i in tqdm(range(param.shape[0])):
                for j in range(param.shape[1]):
                    param[i, j] += h
                    E1 = self.estimate_energy()
                    param[i, j] -= 2 * h
                    E2 = self.estimate_energy()
                    param[i, j] += h
                    grad = (E1 - E2) / (2 * h)
                    grad_array[i, j] = grad

            grad_list.append(grad_array)

        return grad_list

    def analytical_grad(self):
        grad_list = []
        omega = self.omega(self.all_states)
        wf = self.wave_function()
        H = self.hamiltonian

        # loop over b, c and W
        for i, O in enumerate(omega):
            EO = wf.conj().T @ H @ O @ wf
            E = wf.conj().T @ H @ wf
            O = wf.conj().T @ O @ wf
            grad = 2 * (EO - E * O)
            # reshape according to b, c or W
            if i == 0 or i == 1:
                grad = grad.reshape(-1, 1)
            elif i == 2 or i == 3:
                grad = grad.reshape(1, -1)
            else:
                grad = grad.reshape(self.visible_size, self.hidden_size)

            grad_list.append(grad.real)

        return grad_list

    def analytical_estimate_grad(self):
        walker = Walker()
        states = walker(self.unnormalized_probability, num_steps=1000)
        omega_list = self.omega_estiamte(states)
        local_energies = self.local_energy(states)

        grad_list = []

        for omega in omega_list:
            omega_bar = np.mean(omega, axis=0)
            grad = np.mean(np.conj(local_energies)*(omega - omega_bar), axis=0).real *2
            grad_list.append(grad)

        return grad_list

    @utils.timing
    def train(self, iter=100, lr=0.01, analytical_grad=True, print_energy=False):
        energy_list = []
        for i in range(iter):
            if analytical_grad:
                grad_list = self.analytical_estimate_grad()
            else:
                grad_list = self.finite_grad()
            grad_list = self.adam.step(grad_list)
            for param, grad in zip(self.params, grad_list):
                param -= lr * grad
            energy_list.append(self.exact_energy()[0, 0])
            if print_energy:
                print(energy_list[-1])

        return energy_list
