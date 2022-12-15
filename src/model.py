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
            distribution = self.walker.get_history()
        else:
            distribution = dist

        energy = 0

        for state in distribution:
            energy += self.local_energy(state)
        result = energy / len(distribution)

        return np.real(result)

    def finite_difference(self, index):

        params = self.rbm.get_parameters_as_array()
        h = 1/np.sqrt(self.data["walker_steps"])

        params[index] += h
        self.rbm.set_parameters_from_array(params)
        re_plus = self.estimate_energy()

        params[index] -= 2*h
        self.rbm.set_parameters_from_array(params)
        re_minus = self.estimate_energy()

        params[index] += h

        params[index] += h * 1j
        self.rbm.set_parameters_from_array(params)
        im_plus = self.estimate_energy()

        params[index] -= 2 * h * 1j
        self.rbm.set_parameters_from_array(params)
        im_minus = self.estimate_energy()

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

        for i in range(n_steps):
            print(f"Gradient descent step {i}, energy: {self.estimate_energy()}")
            params = params + learning_rate * np.array(self.get_parameter_derivative())
            self.rbm.set_parameters_from_array(params)


class adam(object):