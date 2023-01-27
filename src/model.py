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
            self.walker.estimate_distribution(self.rbm.probability)
            distribution = self.walker.get_history()
        else:
            distribution = dist

        energy = 0

        for state in distribution:

            energy += self.local_energy(state)
        result = energy / len(distribution)

        return np.real(result)

    def exact_energy(self):
        states_list = [utils.int_to_binary_array(i, self.rbm.visible_size) for i in
                       range(2 ** self.rbm.visible_size)]
        local_energy_list = [self.local_energy(state) for state in states_list] # Can be computed only once

        result_list = np.asarray([self.rbm.probability(state) for state in states_list])
        norm = sum(result_list)

        probability_list = result_list / norm

        return sum(local_energy_list * probability_list)

    def get_distribution(self):
        states_list = [utils.int_to_binary_array(i, self.rbm.visible_size) for i in
                       range(2 ** self.rbm.visible_size)]
        result_list = np.asarray([self.rbm.probability(state) for state in states_list])
        norm = sum(result_list)

        return result_list / norm

    def finite_difference(self, index):

        params = self.rbm.get_parameters_as_array()
        h = 1e-3#np.sqrt(self.data["walker_steps"])

        params[index] += h
        self.rbm.set_parameters_from_array(params)
        re_plus = self.exact_energy()
        #re_plus = self.estimate_energy()

        params[index] -= 2*h
        self.rbm.set_parameters_from_array(params)
        re_minus = self.exact_energy()
        #re_minus = self.estimate_energy()
        params[index] += h

        params[index] += h * 1j
        self.rbm.set_parameters_from_array(params)
        im_plus = self.exact_energy()
        #im_plus = self.estimate_energy()

        params[index] -= 2 * h * 1j
        self.rbm.set_parameters_from_array(params)
        im_minus = self.exact_energy()
        #im_minus = self.estimate_energy()

        params[index] += h * 1j

        return (re_plus - re_minus) / (2*h) + (im_plus - im_minus) / (2*h*1j)

    def get_parameter_derivative(self):
        params = self.rbm.get_parameters_as_array()
        params_deriv = []

        for i in range(len(params)):
            params_deriv.append(self.finite_difference(i))

        return params_deriv

    def gradient_descent(self):
        learning_rate = self.data['learning_rate']
        n_steps = self.data['gradient_descent_steps']
        params = self.rbm.get_parameters_as_array()
        adam = Adam()

        for i in range(n_steps):
            try:
                print(f"Gradient descent step {i}, energy: {self.exact_energy()}")
                print(f"Largest param value: {np.max(self.rbm.get_parameters_as_array())}")
                params = params - learning_rate * np.array(self.get_parameter_derivative())
                adam.set_grads(params)
                adam_params = adam.adam_step()
                # print(f"Adam optimized grads: {adam_params}")

                self.rbm.set_parameters_from_array(adam_params)

            except KeyboardInterrupt:
                print("Gradient descent interrupted")
                break


class Adam(object):

    def __init__(self, grads: np.ndarray = None, beta1: float = 0.9, beta2: float = 0.999):
        self.grads = grads
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.t = 0
        self.m = 0
        self.v = 0

    def __call__(self, grads):
        self.grads = grads
        return self.adam_step()

    def set_grads(self, grads):
        self.grads = grads

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
