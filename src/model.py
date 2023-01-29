import copy
from typing import List

from config_parser import get_config_file
from src.ansatz import RBM
from src.mcmc import Walker
import src.utils as utils
import numpy as np


class Model(object):

    def __init__(self, rbm: RBM, walker: Walker, hamiltonian: np.ndarray):

        self.rbm = rbm
        self.walker = walker
        self.hamiltonian = hamiltonian

        self.data = get_config_file()['parameters']  # Load the config file

    def local_energy(self, state: np.ndarray) -> float:
        """Calculates the local energy of the RBM in state s"""

        h_size = self.hamiltonian.shape[0]

        i = utils.binary_array_to_int(state)
        local_state = state
        local_energy = 0
        p_i = self.rbm.amplitude(local_state)

        for j in range(h_size):
            p_j = self.rbm.amplitude(utils.int_to_binary_array(j, state.size))

            h_ij = self.hamiltonian[i, j]

            local_energy += h_ij * p_j / p_i

        return local_energy

    def estimate_energy(self, dist: List[np.ndarray] = None) -> float:
        if dist is None:
            distribution = self.get_mcmc_states()
        else:
            distribution = dist
        energy = 0

        for state in distribution:
            energy += self.local_energy(state)
        result = energy / len(distribution)

        return np.real(result)

    def get_all_states(self):
        return np.asarray([utils.int_to_binary_array(i, self.rbm.visible_size)
                           for i in range(2 ** self.rbm.visible_size)])

    def get_mcmc_states(self):
        self.walker.estimate_distribution(self.rbm.probability)
        return np.asarray(self.walker.get_history())

    def get_prob_distribution(self):
        result_list = np.asarray([self.rbm.probability(state) for state in self.get_all_states()])
        norm = sum(result_list)

        return result_list / norm

    def exact_energy(self) -> float:
        # Calculates the exact energy of the model by sampling over all possible states
        local_energy_list = [self.local_energy(state) for state in self.get_all_states()]  # Can be computed only once
        probability_list = self.get_prob_distribution()

        return sum(probability_list * local_energy_list)

    def finite_difference_step(self, index, param, exact_dist, h=1e-2):

        if not exact_dist:
            function = self.estimate_energy
        else:
            function = self.exact_energy
        reset_value = param

        param += h
        self.rbm.set_parameter_from_value(index, param)
        re_plus = function()

        param -= 2*h
        self.rbm.set_parameter_from_value(index, param)
        re_minus = function()

        param += h

        param += h * 1j
        self.rbm.set_parameter_from_value(index, param)
        im_plus = function()

        param -= 2 * h * 1j
        self.rbm.set_parameter_from_value(index, param)
        im_minus = function()

        self.rbm.set_parameter_from_value(index, reset_value)

        return (re_plus - re_minus) / (2*h) + (im_plus - im_minus) / (2*h*1j)

    def finite_difference(self, params, exact_dist):
        gradients = []

        for i, param in enumerate(params):
            gradients.append(self.finite_difference_step(i, param, exact_dist))

        return np.asarray(gradients, dtype=complex)

    def gradient_descent(self,
                         gradient_method=None,
                         learning_rate=None,
                         n_steps=None,
                         termination_condition=None,
                         adam_optimization=None,
                         exact_dist=None):

        if gradient_method is None or gradient_method == 'analytical':
            gradient = self.analytical_gradients
        elif gradient_method == 'finite_difference':
            gradient = self.finite_difference
        else:
            raise ValueError("Gradient method not recognized")

        if exact_dist is None:
            exact_dist = self.data['exact_distribution']
        else:
            exact_dist = exact_dist

        if learning_rate is None:
            learning_rate = self.data['learning_rate']
        else:
            learning_rate = learning_rate

        if n_steps is None:
            n_steps = self.data['gradient_descent_steps']
        else:
            n_steps = n_steps

        if termination_condition is None:
            termination_condition = self.data['termination_threshold']
        else:
            termination_condition = termination_condition

        if adam_optimization is None:
            adam_optimization = self.data['adam_optimizer']
        else:
            adam_optimization = adam_optimization

        params = copy.deepcopy(self.rbm.get_parameters_as_array())

        if adam_optimization:
            adam = Adam()

        energy_landscape = []
        a = 0

        for i in range(n_steps):
            try:
                b = a

                energy = self.exact_energy()
                a = energy
                print(f"Gradient descent step {i + 1}, energy: {energy}")
                energy_landscape.append(energy)

                if adam_optimization:
                    new_grads = adam(gradient(params, exact_dist))
                else:
                    new_grads = gradient(params, exact_dist)
                    
                params = params - learning_rate * np.array(new_grads)
                self.rbm.set_parameters_from_array(params)

            except KeyboardInterrupt:
                print("Gradient descent interrupted")
                break

            if termination_condition > abs(b-a):
                print("Termination condition reached")
                break

        return energy_landscape

    def analytical_gradients(self, params, exact_dist):
        if exact_dist:
            gradient = self.exact_grads
        else:
            gradient = self.estimate_grads

        return gradient(params)

    def exact_grads(self, params):
        distribution = self.get_all_states()
        g_j = np.zeros(len(params), dtype=complex)
        omega_j = []

        for i, state in enumerate(distribution):
            omega_j.append(self.omega(state))

        omega_j = np.transpose(np.asarray(omega_j))

        for j in range(len(params)):
            g_j[j] = min(np.linalg.eigvalsh(self.hamiltonian * np.diag(omega_j[j]))) \
                     - self.exact_energy() * min(np.linalg.eigvalsh(np.diag(omega_j[j])))
        grads = 2 * np.real(g_j)

        return grads

    def estimate_grads(self, params) -> np.ndarray:
        distribution = self.get_mcmc_states()
        omega_bar = self.omega_bar(distribution)

        grads = np.zeros(len(params))

        for j in range(len(params)):
            g_j = 0
            for state in distribution:
                g_j += np.conjugate(self.local_energy(state)) * (self.omega(state) - omega_bar)

            grads = 2 * np.real(g_j)/len(distribution)

        return grads

    def omega_bar(self, dist: np.ndarray = None) -> np.ndarray:
        """
        Calculates the average of the omega function over the distribution
        :param dist:
        :return: ndarray with the average of the omega function for each parameter

        """

        if dist is None:
            self.walker.estimate_distribution(self.rbm.probability)
            distribution = self.walker.get_history()
        else:
            distribution = dist

        omega_bar_j = np.zeros(len(self.rbm.get_parameters_as_array()))
        omega_j = 0

        for j in range(len(self.rbm.get_parameters_as_array())):
            for state in distribution:
                omega_j += self.omega(state)

            omega_bar_j = omega_j / len(distribution)

        return omega_bar_j

    def omega(self, state) -> np.ndarray:

        return 1/self.rbm.amplitude(state) * self.param_grads(state)

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
        _1 = state @ -(self.rbm.W + self.rbm.c)
        _2 = np.exp(_1)
        hidden_bias_grads = -(_2 / (1 + _2))

        return np.asarray(hidden_bias_grads, dtype=complex)

    def _weights_grads(self, state) -> np.ndarray:
        _1 = state @ -(self.rbm.W + self.rbm.c)
        _2 = np.exp(_1)/(1 + np.exp(_1))
        weight_gradients = -1 * _2.reshape(-1, 1) @ state.reshape(1, -1)

        return np.asarray(weight_gradients, dtype=complex)

    def plot_mcmc_vs_exact(self):
        from matplotlib import pyplot as plt
        self.walker.estimate_distribution(self.rbm.probability)
        history = [utils.binary_array_to_int(state) for state in self.walker.get_history()]

        plt.figure(0)
        plt.hist(history, density=True, bins=range(2 ** self.rbm.visible_size + 1), edgecolor="black", align='left',
                 rwidth=0.8)
        plt.scatter([x for x in range(2 ** self.rbm.visible_size)], self.get_prob_distribution(), color='red')
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

    #def set_grads(self, grads):
    #    self.grads = grads

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
