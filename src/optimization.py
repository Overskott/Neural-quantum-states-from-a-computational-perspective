import numpy as np

from config_parser import get_config_file
import src.utils as utils


class FiniteDifference(object):

    def __init__(self):

        self.data = get_config_file()['parameters']  # Load the config file
        self.model = None
        self.exact_dist = None

    def __call__(self, model, exact_dist=None):
        self.model = model

        if exact_dist is None:
            self.exact_dist = self.data['exact_distribution']
        else:
            self.exact_dist = exact_dist

        return self.finite_difference()

    def finite_difference(self):
        gradients = np.zeros(len(self.model.rbm.get_parameters_as_array()))

        for i, param in enumerate(self.model.rbm.get_parameters_as_array()):
            gradients[i] = self.finite_difference_step(i, param, exact_dist=self.exact_dist)

        return gradients

    def finite_difference_step(self, index, param, exact_dist, h=1e-4):

        if not exact_dist:
            function = self.model.estimate_energy

        else:
            function = self.model.exact_energy

        reset_value = param

        param += h
        self.model.rbm.set_parameter_from_value(index, param)
        re_plus = function()

        param -= 2 * h
        self.model.rbm.set_parameter_from_value(index, param)
        re_minus = function()

        self.model.rbm.set_parameter_from_value(index, reset_value)

        return (re_plus - re_minus) / (2 * h)


class AnalyticalGradient(object):

    def __init__(self):

        self.data = get_config_file()['parameters']  # Load the config file
        self.model = None
        self.exact_dist = None

    def __call__(self, model, exact_dist=None):
        self.model = model

        if exact_dist is None:
            self.exact_dist = self.data['exact_distribution']
        else:
            self.exact_dist = exact_dist

        return self.analytical_gradients()

    def analytical_gradients(self):
        if self.exact_dist:
            gradient = self.exact_grads
        else:
            gradient = self.estimate_grads

        return gradient(self.model.rbm.get_parameters_as_array())

    def exact_grads(self, params):
        distribution = self.model.get_all_states()
        g_j = np.zeros(len(params), dtype=complex)
        omega_j = []

        for i, state in enumerate(distribution):
            omega_j.append(self.omega_exact(state))

        omega_j = np.transpose(np.asarray(omega_j))

        wave_function = np.asarray(self.model.get_wave_function())

        energy = self.model.exact_energy(distribution)

        for j in range(len(params)):
            g_j[j] = wave_function.conj() @ (self.model.hamiltonian @ np.diag(omega_j[j])) @ wave_function - \
                     (energy * wave_function.conj() @ np.diag(omega_j[j]) @ wave_function)

        grads = 2 * np.real(g_j)

        return grads

    def estimate_grads(self, params) -> np.ndarray:
        distribution = self.model.get_mcmc_states()
        omega_bar = self.omega_bar(distribution)

        g_j = 0
        for state in distribution:
            g_j += np.conjugate(self.model.local_energy(state)) * (self.omega(state) - omega_bar)

        grads = 2 * np.real(g_j) / len(distribution)

        return grads

    def omega_bar(self, dist: np.ndarray = None) -> np.ndarray:
        """
        Calculates the average of the omega function over the distribution
        :param dist:
        :return: ndarray with the average of the omega function for each parameter

        """

        if dist is None:
            distribution = self.model.etimate_distribution()
        else:
            distribution = dist

        omega_j = np.zeros(len(self.model.rbm.get_parameters_as_array()), dtype=complex)
        for state in distribution:
            omega_j += self.omega(state)

        omega_bar_j = omega_j / len(distribution)

        return omega_bar_j

    def omega(self, state) -> np.ndarray:
        return 1 / self.model.rbm.amplitude(state) * self.param_grads(state)

    def omega_exact(self, state) -> np.ndarray:
        return 1 / self.model.get_amplitude_normalized(state) * self.param_grads(state)

    def param_grads(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the analytical gradient of the parameters with respect to the energy of the state.

            Args:
                state (np.ndarray): The state of the RBM to calculate the gradient for.

            Returns:
                np.ndarray: The gradient of the parameters with respect to the energy of the state.
        """
        b_grad_r = b_grad_i = self._visible_bias_grads(state)
        c_grad_r = self._hidden_bias_grads_r(state)
        c_grad_i = self._hidden_bias_grads_i(state)
        w_grad_r = self._weights_grads_r(state)
        w_grad_i = self._weights_grads_i(state)

        return np.concatenate((b_grad_r, c_grad_r, w_grad_r.flatten(), b_grad_i, c_grad_i, w_grad_i.flatten()))

    def _visible_bias_grads(self, state) -> np.ndarray:
        """
        Calculates the gradient of the visible bias with respect to the energy of the state.

            Args:
                state (np.ndarray): The state of the RBM to calculate the visible bias gradient for.

            Returns:
                np.ndarray: The gradient of the visible bias with respect to the energy of the state.
        """
        return -1 * state

    def _hidden_bias_grads_r(self, state) -> np.ndarray:

        exponent_r = -(state @ self.model.rbm.W_r + self.model.rbm.c_r)
        expression_r = np.exp(exponent_r)
        hidden_bias_grads_r = -(expression_r / (1 + expression_r))

        return np.asarray(hidden_bias_grads_r, dtype=complex)

    def _hidden_bias_grads_i(self, state) -> np.ndarray:
        exponent_i = -(state @ self.model.rbm.W_i + self.model.rbm.c_i)
        expression_i = np.exp(exponent_i)
        hidden_bias_grads_i = -(expression_i / (1 + expression_i))

        return np.asarray(hidden_bias_grads_i, dtype=complex)

    def _weights_grads_r(self, state) -> np.ndarray:
        _1 = -(state @ self.model.rbm.W_r + self.model.rbm.c_r)
        _2 = np.exp(_1) / (1 + np.exp(_1))
        weight_gradients = -1 * _2.reshape(-1, 1) @ state.reshape(1, -1)

        return np.asarray(weight_gradients, dtype=complex)

    def _weights_grads_i(self, state) -> np.ndarray:
        _1 = -(state @ self.model.rbm.W_i + self.model.rbm.c_i)
        _2 = np.exp(_1) / (1 + np.exp(_1))
        weight_gradients = -1 * _2.reshape(-1, 1) @ state.reshape(1, -1)

        return np.asarray(weight_gradients, dtype=complex)

    def plot_mcmc_vs_exact(self):
        from matplotlib import pyplot as plt

        history = [utils.binary_array_to_int(state) for state in self.model.get_mcmc_states()]

        plt.figure(0)
        plt.hist(history, density=True, bins=range(2 ** self.model.rbm.visible_size + 1), edgecolor="black", align='left',
                 rwidth=0.8)
        plt.scatter([x for x in range(2 ** self.model.rbm.visible_size)], self.model.get_prob_distribution(), color='red')
        plt.title("RBM Probability Distribution")
        plt.xlabel('State')
        plt.ylabel('Probalility')
        plt.legend(['Analytic Results', 'MCMC Results'])

        plt.show()


class Adam(object):

    def __init__(self, grads=None, beta1: float = 0.9, beta2: float = 0.999):
        self.grads = grads
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.t = 0
        self.m = 0
        self.v = 0

    def __call__(self, grads):
        if grads is None:
            print("Missing argument: grads in Adam")

        self.grads = grads
        return self.adam_step()

    def adam_step(self):
        self.t += 1
        weight_gradient_modified = self.optimize_grads()

        return weight_gradient_modified

    def optimize_grads(self):
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.abs(self.grads) ** 2

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        adam_grad = m_hat / (np.sqrt(v_hat) + self.epsilon)

        return adam_grad
