from src.ansatz import RBM
from src.mcmc import Walker
import src.utils as utils
import numpy as np


class Model(object):

    def __init__(self, rbm: RBM, walker: Walker, hamiltonian: np.ndarray):

        self.rbm = rbm
        self.walker = walker
        self.hamiltonian = hamiltonian

    def local_energy(self, state: np.ndarray) -> float:
        """Calculates the local energy of the RBM in state s"""

        h_size = self.hamiltonian.shape[0]
        i = state
        local_state = state
        local_energy = 0

        for j in range(h_size):
            p_i = self.rbm.probability(local_state)
            p_j = self.rbm.probability(utils.int_to_binary_array(j, state.size))

            h_ij = self.hamiltonian[i, j]

            local_energy += h_ij * p_j/p_i

        return local_energy

    def estimate_energy(self):
        distribution = self.walker.get_history()
        energy = 0
        for state in distribution:
            energy += self.local_energy(state)

        return energy / len(distribution)

    def finite_difference(self, func, x, h=1e-5):
        #return (func(x + h) - func(x - h)) / (2 * h)
        pass

    def get_parameter_derivative(self, func, h=1e-5):
        params = self.rbm.get_parameters_as_array()
        params_deriv = []

        for param in params:
            params_deriv.append(self.finite_difference(func, param, h))

    def gradient_descent(self, func, params, learning_rate=0.01, n_steps=1000):
        params = np.array(params)
        for i in range(n_steps):
            params = params - learning_rate * np.array(self.get_parameter_derivative(func))
        return params


