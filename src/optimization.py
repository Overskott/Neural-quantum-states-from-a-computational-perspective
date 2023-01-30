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
        gradients = []

        for i, param in enumerate(self.model.rbm.get_parameters_as_array()):
            gradients.append(self.finite_difference_step(i, param, exact_dist=self.exact_dist))

        return np.asarray(gradients, dtype=complex)

    def finite_difference_step(self, index, param, exact_dist, h=1e-2):

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

        param += h

        param += h * 1j
        self.model.rbm.set_parameter_from_value(index, param)
        im_plus = function()

        param -= 2 * h * 1j
        self.model.rbm.set_parameter_from_value(index, param)
        im_minus = function()

        self.model.rbm.set_parameter_from_value(index, reset_value)

        return (re_plus - re_minus) / (2 * h) + (im_plus - im_minus) / (2 * h * 1j)


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
            omega_j.append(self.omega(state))

        omega_j = np.transpose(np.asarray(omega_j))

        for j in range(len(params)):
            g_j[j] = min(np.linalg.eigvalsh(self.model.hamiltonian * np.diag(omega_j[j]))) \
                     - self.model.exact_energy() * min(np.linalg.eigvalsh(np.diag(omega_j[j])))
        grads = 2 * np.real(g_j)

        return grads

    def estimate_grads(self, params) -> np.ndarray:
        distribution = self.model.get_mcmc_states()
        omega_bar = self.omega_bar(distribution)

        grads = np.zeros(len(params))

        for j in range(len(params)):
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
            self.model.walker.estimate_distribution(self.model.rbm.probability)
            distribution = self.model.walker.get_history()
        else:
            distribution = dist

        omega_bar_j = np.zeros(len(self.model.rbm.get_parameters_as_array()))
        omega_j = 0

        for j in range(omega_bar_j.shape[0]):
            for state in distribution:
                omega_j += self.omega(state)

            omega_bar_j = omega_j / len(distribution)

        return omega_bar_j

    def omega(self, state) -> np.ndarray:

        return 1 / self.model.rbm.amplitude(state) * self.param_grads(state)

    def param_grads(self, state: np.ndarray) -> np.ndarray:
        """
        Calculates the analytical gradient of the parameters with respect to the energy of the state.

            Args:
                state (np.ndarray): The state of the RBM to calculate the gradient for.

            Returns:
                np.ndarray: The gradient of the parameters with respect to the energy of the state.
        """
        b_grad = self._visible_bias_grads(state)
        c_grad = self._hidden_bias_grads(state)
        w_grad = self._weights_grads(state)

        return np.concatenate((b_grad, c_grad, w_grad.flatten()))

    def _visible_bias_grads(self, state) -> np.ndarray:
        """
        Calculates the gradient of the visible bias with respect to the energy of the state.

            Args:
                state (np.ndarray): The state of the RBM to calculate the visible bias gradient for.

            Returns:
                np.ndarray: The gradient of the visible bias with respect to the energy of the state.
        """
        return -1 * state

    def _hidden_bias_grads(self, state) -> np.ndarray:
        _1 = state @ -(self.model.rbm.W + self.model.rbm.c)
        _2 = np.exp(_1)
        hidden_bias_grads = -(_2 / (1 + _2))

        return np.asarray(hidden_bias_grads, dtype=complex)

    def _weights_grads(self, state) -> np.ndarray:
        _1 = state @ -(self.model.rbm.W + self.model.rbm.c)
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
