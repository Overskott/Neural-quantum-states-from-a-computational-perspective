import copy
from typing import List

from config_parser import get_config_file
from src.ansatz import RBM
from src.mcmc import Walker
import src.utils as utils
import numpy as np

from src.optimization import Adam, FiniteDifference, AnalyticalGradient


class Model(object):

    def __init__(self, rbm: RBM, walker: Walker, hamiltonian: np.ndarray):

        self.rbm = rbm
        self.walker = walker
        self.hamiltonian = hamiltonian

        self.data = get_config_file()['parameters']  # Load the config file

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

    def gradient_descent(self,
                         gradient_method=None,
                         learning_rate=None,
                         n_steps=None,
                         termination_condition=None,
                         adam_optimization=None,
                         exact_dist=None):

        if gradient_method is None or gradient_method == 'analytical':
            gradient = AnalyticalGradient()
        elif gradient_method == 'finite_difference':
            gradient = FiniteDifference()
        else:
            raise ValueError("Gradient method not recognized")

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

        if adam_optimization:
            adam = Adam()

        params = copy.deepcopy(self.rbm.get_parameters_as_array())
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
                    new_grads = adam(gradient(self, exact_dist))
                else:
                    new_grads = gradient(self, exact_dist)

                params = params - learning_rate * np.array(new_grads)

                self.rbm.set_parameters_from_array(params)

            except KeyboardInterrupt:
                print("Gradient descent interrupted")
                break

            if termination_condition > abs(b - a):
                print("Termination condition reached")
                break

        return energy_landscape

