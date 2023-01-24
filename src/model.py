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
            distribution = self.get_distribution()
        else:
            distribution = dist
        energy = 0

        for state in distribution:
            energy += self.local_energy(state)
        result = energy / len(distribution)

        return np.real(result)

    def exact_energy(self) -> float:
        # Calculates the exact energy of the model by sampling over all possible states

        states_list = [utils.int_to_binary_array(i, self.rbm.visible_size) for i in
                       range(2 ** self.rbm.visible_size)]
        local_energy_list = [self.local_energy(state) for state in states_list] # Can be computed only once

        result_list = np.asarray([self.rbm.probability(state) for state in states_list])
        norm = sum(result_list)

        probability_list = result_list / norm

        return sum(local_energy_list * probability_list)

    def analytical_params_grad(self, distribution: List[np.ndarray]) -> np.ndarray:
        omega_bar = self.omega_bar(distribution)
        grads = np.zeros(len(self.rbm.get_parameters_as_array()))

        for j in range(len(self.rbm.get_parameters_as_array())):
            g_j = 0
            for state in distribution:
                g_j += np.conjugate(self.local_energy(state)) * (self.omega(state) - omega_bar) # TODO do we need complex conjugate here?

            grads = 2 * np.real(g_j)/len(distribution)


        return grads

    def omega_bar(self, dist: List[np.ndarray] = None) -> np.ndarray:
        '''
        Calculates the average of the omega function over the distribution
        :param dist:
        :return: ndarray with the average of the omega function for each parameter

        '''
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

        b_grad = self.visible_bias_grads(state)
        #print(f"b_grad: {b_grad}")
        c_grad = self.hidden_bias_grads(state)
        #print(f"c_grad: {c_grad}")
        w_grad = self.weights_grads(state)
        #print(f"w_grad: {w_grad}")

        #print(f"Grads: {np.concatenate((b_grad, c_grad, w_grad.flatten()))}")
        return np.concatenate((b_grad, c_grad, w_grad.flatten()))

    def visible_bias_grads(self, state) -> np.ndarray:
        return -1 * state

    def hidden_bias_grads(self, state) -> np.ndarray:
        hidden_bias_grads = np.zeros(self.rbm.hidden_size, dtype=complex)

        for j in range(self.rbm.hidden_size):
            _1 = -self.rbm.W[:, j] @ state - self.rbm.c[j]
            _2 = np.exp(_1)
            hidden_bias_grads[j] = -(_2 / (1 + _2))

        return hidden_bias_grads

    def weights_grads(self, state) -> np.ndarray:

        weight_gradients = np.zeros((self.rbm.visible_size, self.rbm.hidden_size), dtype=complex)

        for i in range(self.rbm.hidden_size): #TODO is i and j correct?
            _1 = (-self.rbm.W[:, i] @ state) - self.rbm.c[i]
            _2 = np.exp(_1)/(1 + np.exp(_1))

            for j in range(self.rbm.visible_size):

                weight_gradients[j, i] = -state[j] * _2 #TODO check minus error in gradients (should be minus here)

        return weight_gradients

    def get_prob_distribution(self):
        states_list = [utils.int_to_binary_array(i, self.rbm.visible_size) for i in
                       range(2 ** self.rbm.visible_size)]
        result_list = np.asarray([self.rbm.probability(state) for state in states_list])
        norm = sum(result_list)

        return result_list / norm

    def get_distribution(self):
        self.walker.estimate_distribution(self.rbm.probability)
        return self.walker.get_history()

    def get_exact_distribution(self):
        states_list = [utils.int_to_binary_array(i, self.rbm.visible_size) for i in
                       range(2 ** self.rbm.visible_size)]
        result_list = np.asarray([self.rbm.probability(state) for state in states_list])
        return result_list


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

    def gradient_descent_1(self):
        learning_rate = self.data['learning_rate']
        n_steps = self.data['gradient_descent_steps']
        params = self.rbm.get_parameters_as_array()
        adam = Adam()

        for i in range(n_steps):
            try:
                print(f"Gradient descent step {i}, energy: {self.exact_energy()}")
                #print(f"Largest param value: {np.max(self.rbm.get_parameters_as_array())}")



                params = params - learning_rate * np.array(self.get_parameter_derivative())
                print(f"Params 1: {params}")

                adam.set_grads(params)
                adam_params = adam.adam_step()
                # print(f"Adam optimized grads: {adam_params}")

                self.rbm.set_parameters_from_array(adam_params)

            except KeyboardInterrupt:
                print("Gradient descent interrupted")
                break

    def gradient_descent_2(self):
        learning_rate = self.data['learning_rate']
        n_steps = self.data['gradient_descent_steps']
        adam = Adam()
        #dist = self.get_distribution()

        for i in range(n_steps):
            try:
                print(f"Gradient descent step {i}, energy: {self.exact_energy()}")
                #print(f"Largest param value: {np.max(self.rbm.get_parameters_as_array())}")
                params = self.rbm.get_parameters_as_array()
                #print(f"Params: {params}")
                #print(f"abs grads: {self.analytical_params_grad(dist)}")
                dist = self.get_distribution()

                adam_grads = adam(self.analytical_params_grad(dist))
                #adam.set_grads(self.analytical_params_grad(dist))
                #adam_grads = adam.adam_step()
                params = params + learning_rate * np.array(adam_grads)
                print(f"Params 2: {params}")

                # print(f"Adam optimized grads: {adam_params}")

                self.rbm.set_parameters_from_array(params)

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
