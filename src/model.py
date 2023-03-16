import copy
import time
from typing import List

from config_parser import get_config_file
from src.ansatz import RBM
from src.hamiltonians import Hamiltonian, IsingHamiltonian, ReducedIsingHamiltonian, DiagonalHamiltonian
from src.mcmc import Walker
import src.utils as utils
import numpy as np

from src.optimization import Adam, FiniteDifference, AnalyticalGradient


class Model(object):

    def __init__(self, rbm: RBM = None, walker: Walker = None, hamiltonian: Hamiltonian = None):

        self.rbm = rbm
        self.walker = walker
        self.hamiltonian = hamiltonian
        #self.off_diag = utils.get_matrix_off_diag_range(self.hamiltonian)
        self.data = get_config_file()['parameters']  # Load the config file

        self.optimizing_time = 0

    def get_all_states(self):
        return np.asarray([utils.int_to_binary_array(i, self.rbm.visible_size)
                           for i in range(2 ** self.rbm.visible_size)])

    def get_mcmc_states(self):
        self.walker.estimate_distribution(self.rbm.probability)
        return np.asarray(self.walker.get_history())

    def get_prob_distribution(self, dist):
        result_list = np.asarray([self.rbm.probability(state) for state in dist])
        norm = sum(result_list)

        return result_list / norm

    def get_wave_function(self):
        dist = self.get_all_states()
        amp_list = self.rbm.amplitude(dist)
        norm = np.sqrt(sum(np.abs(amp_list)**2))

        return amp_list / norm

    def get_amplitude_normalized(self, state):
        wf = self.get_wave_function()
        amp = wf[utils.binary_array_to_int(state)]

        return amp

    def exact_energy(self, dist=None) -> float:

        if dist is None:
            distribution = self.get_all_states()
        else:
            distribution = dist
        # Calculates the exact energy of the model by sampling over all possible states
        local_energy_list = np.asarray([self.local_energy(state) for state in distribution])  # Can be computed only once
        probability_list = self.get_prob_distribution(distribution)

        return np.real(sum(probability_list * local_energy_list))

    #@profile
    def estimate_energy(self, dist: List[np.ndarray] = None) -> float:
        if dist is None:
            distribution = self.get_mcmc_states()
        else:
            distribution = dist
        #energy = 0

        #for state in distribution:

           # energy += self.local_energy(state)

        #result = energy / len(distribution)

        if type(self.hamiltonian) == IsingHamiltonian:
            result = np.linalg.eig(self.hamiltonian)[1].T[0]
            return np.real(result)
        elif type(self.hamiltonian) == ReducedIsingHamiltonian:
            result = sum([self.ising_local_energy(state) for state in distribution]) / len(distribution)
            return np.real(result)
        elif type(self.hamiltonian) == DiagonalHamiltonian:
            result = sum([self.local_energy(state) for state in distribution]) / len(distribution)
            return np.real(result)

        else:
            result = sum([self.local_energy(state) for state in distribution]) / len(distribution)

            return np.real(result)

    #@profile
    def local_energy_fast(self, dist: np.ndarray) -> complex:
        """Calculates the local energy of the RBM in state s"""

        i = [utils.binary_array_to_int(state) for state in dist]
        j = self.get_all_states()
        p_i = self.rbm.amplitude(dist)
        p_j = self.rbm.amplitude(j)

        for index, onehot in enumerate(i):
            el_i = H[:] * p_i[:index]

        local_energy = 0

        H = self.hamiltonian
        hamiltonian_size = H.shape[0]

        def eloc_index_value(index_1, index_2):

            i = [utils.binary_array_to_int(state) for state in dist]
            p_i = self.rbm.amplitude(dist)

            j = np.asarray([utils.int_to_binary_array(j, self.rbm.visible_size)for j
                            in range(hamiltonian_size)])
            p_j = self.rbm.amplitude(j)

            h_ij = self.hamiltonian[index_1, index_2]
            return h_ij * p_j / p_i

        if type(self.hamiltonian) == DiagonalHamiltonian:
            for j in range(self.hamiltonian.diagonal(), - self.hamiltonian.diagonal() - 1, -1):
                j = i + j

                if j < 0 or j >= hamiltonian_size-1:  # If the j index is out of bounds skip calculation
                    continue
                else:
                    local_energy += eloc_index_value(i, j)
        else:
            local_energy = sum([eloc_index_value(i, j) for j in range(hamiltonian_size)])

        return local_energy

    @DeprecationWarning
    def is_diagonal(self):
        if self.hamiltonian.diagonal() + 1 - self.hamiltonian.size == 0:
            return True
        else:
            return False

    def ising_local_energy(self, state: np.ndarray) -> complex:
        gamma = self.hamiltonian
        p_i = self.rbm.amplitude(state)

        def create_mu(state, index):
            mu = state.copy()

            mu[index] = 1 - state[index]

            if index + 1 == len(state):
                pass
            else:
                mu[index+1] = 1 - state[index+1]
            return mu

        def eloc_index_value(gamma, index_1):
            mu_i = create_mu(state, index_1)
            p_j = self.rbm.amplitude(mu_i)
            return gamma * p_j / p_i

        local_energy = sum([eloc_index_value(g, j) for (j, g) in enumerate(gamma)])

        return local_energy

    #@profile
    def gradient_descent(self,
                         gradient_method=None,
                         learning_rate=None,
                         n_steps=None,
                         termination_condition=None,
                         adam_optimization=None,
                         exact_dist=None,
                         h_type='generic'):

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

        start = time.time()

        for i in range(n_steps):
            try:
                energy = self.exact_energy()
                print(f"Gradient descent step {i + 1}, energy: {energy}")
                energy_landscape.append(energy)
                #print(f"Gradient: {gradient(self, exact_dist)}")
                if adam_optimization:
                    new_grads = adam(gradient(self, exact_dist))
                else:
                    new_grads = gradient(self, exact_dist)

                params = params - learning_rate * np.array(new_grads)

                self.rbm.set_parameters_from_array(params)

            except KeyboardInterrupt:
                print("Gradient descent interrupted")
                break

        self.optimizing_time = time.time() - start

        return energy_landscape


class GradientDescent(object):

    def __init__(self, model: Model, learning_rate=None, n_steps=None, termination_condition=None, adam_optimization=None):
        self.model = model
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.termination_condition = termination_condition
        self.adam_optimization = adam_optimization

    def __call__(self, *args, **kwargs):
        return self.model.gradient_descent(self.learning_rate, self.n_steps, self.termination_condition,
                                           self.adam_optimization)

    def __str__(self):
        return f"Gradient descent with learning rate {self.learning_rate}, " \
               f"{self.n_steps} steps and termination condition {self.termination_condition}"

    def __repr__(self):
        return f"Gradient descent with learning rate {self.learning_rate}, " \
               f"{self.n_steps} steps and termination condition {self.termination_condition}"
