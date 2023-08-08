import copy
import line_profiler
profile = line_profiler.LineProfiler()  #Line for handling profiling annotations

import numpy as np
import src.utils as utils
from config_parser import get_config_file


class Hamiltonian(np.lib.mixins.NDArrayOperatorsMixin):
    """
    A  class for representing general Hamiltonians as numpy arrays

    Attributes:
        dim (int): The dimension of the Hamiltonian matrix (dim x dim).
        values (np.ndarray): The values of the Hamiltonian i.e. its matrix elements.

    """

    def __init__(self):
        """
        Constructor for the Hamiltonian class

        :param n: Number of spins/qubits in the system
        :param values: The elements of the Hamiltonian matrix

        """
        self.values = None

    def __repr__(self):
        return f"{self.__class__.__name__}\n({self.values})"

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __array__(self):
        return self.values


class RandomHamiltonian(Hamiltonian):

    def __init__(self, n):
        super().__init__()

        self.values = utils.random_hamiltonian(n)


class IsingHamiltonian(Hamiltonian):

    def __init__(self, n=None, gamma=None):
        super().__init__()

        if gamma is None:
            self.values = utils.random_ising_hamiltonian(n=n)
        else:
            self.values = utils.random_ising_hamiltonian(gamma_array=gamma)


class IsingHamiltonianReduced(Hamiltonian):

    def __init__(self, n=None, gamma=None):
        super().__init__()

        if gamma is None:
            self.values = utils.random_gamma(n)
        else:
            self.values = gamma


class RBM(object):
    """
    A class for representing Restricted Boltzmann Machines (RBMs) as numpy arrays.

    Attributes:
        visible_size (int): The number of visible nodes in the RBM.
        hidden_size (int): The number of hidden nodes in the RBM.
        hamiltonian (Hamiltonian): The Hamiltonian of the system.
        walker_steps (int): The number of walker steps to be performed in the RBM.

        b_r (np.ndarray): The real valued biases of the visible nodes.
        b_i (np.ndarray): The imaginary valued biases of the visible nodes.
        c_r (np.ndarray): The real valued biases of the hidden nodes.
        c_i (np.ndarray): The imaginary valued biases of the hidden nodes.
        W_r (np.ndarray): The real valued weights of the RBM.
        W_i (np.ndarray): The imaginary valued weights of the RBM.

        params (list): A list containing the all the parameters (in order b_r, b_i, c_r, c_i, W_r, W_i) of the RBM.
        all_states (np.ndarray): A list containing all possible states of the RBM.
        adam (Adam): The Adam optimizer.
    """

    def __init__(self,
                 visible_size: int,
                 hidden_size: int,
                 hamiltonian: Hamiltonian,
                 walker_steps: int
                 ):

        """
        Constructor for the RBM class
        :param visible_size: The number of visible nodes in the RBM.
        :param hidden_size: The number of hidden nodes in the RBM.
        :param hamiltonian: The Hamiltonian of the system.
        :param walker_steps: Number of walker steps to be collected by the mcmc sampler.
               Value 0 means that the distribution will be the actual one.
        """

        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.hamiltonian = hamiltonian
        self.walker_steps = walker_steps

        scale = 1  # /np.sqrt(hidden_size)
        self.b_r = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias
        self.b_i = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias

        self.c_r = np.random.normal(0, 1/scale, (1, self.hidden_size))  # Hidden layer bias
        self.c_i = np.random.normal(0, 1/scale, (1, self.hidden_size))  # Hidden layer bias

        self.W_r = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights
        self.W_i = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights

        self.params = [self.b_r, self.b_i, self.c_r, self.c_i, self.W_r, self.W_i]

        # Generate all possible states
        self.all_states = self.get_all_states()

        # initialize the Adam optimizer
        self.adam = Adam()

    def get_all_states(self):
        """
        Generates all possible states of the RBM and returns them as a numpy array.
        :return: Distribution of all possible states as numpy array.
        """
        all_states_list = []

        for i in range(2 ** self.visible_size):
            state = utils.numberToBase(i, 2, self.visible_size)
            all_states_list.append(state)

        return np.array(all_states_list)

    def wave_function(self):
        """
        Calculates the wave function of the RBM by sampling over all states.
        :return: the wave function of the RBM.
        """
        return self.normalized_amplitude(self.all_states)

    def probability_dist(self):
        """
        Calculates the probability distribution of the RBM over all states.
        :return: The probability distribution of the RBM.
        """
        return np.abs(self.wave_function()) ** 2

    def mcmc_dist(self):
        walker = Walker(self.visible_size, self.walker_steps)
        walker(self.probability, self.walker_steps)

        return walker.get_history()

    def mcmc_state_estimate(self):

        mcmc_dist = [utils.binary_array_to_int(state) for state in self.mcmc_dist()]
        index, counts = np.unique(mcmc_dist, return_counts=True)

        prob_vector = np.zeros(2**self.visible_size)
        prob_vector[index] = counts

        return prob_vector/np.sum(prob_vector)

    def normalized_amplitude(self, state):
        """ Calculates and returns the normalized amplitude of the given state """
        all_amps = np.abs(self.unnormalized_amplitude(self.all_states))
        Z = np.sqrt(np.sum(np.abs(all_amps) ** 2))  # The normalization constant
        return self.unnormalized_amplitude(state) / Z

    def probability(self, state: np.ndarray) -> float:
        """
        Calculates and returns the probability of finding the RBM in the given state
        """
        return np.abs(self.normalized_amplitude(state)) ** 2

    def unnormalized_amplitude(self, state):
        Wstate = np.matmul(state, self.W_r) + 1j * np.matmul(state, self.W_i)
        exponent = Wstate + self.c_r + 1j * self.c_i
        A = np.exp(-exponent)
        A = np.prod(1 + A, axis=1, keepdims=True)
        A = A * np.exp(-np.matmul(state, self.b_r) - 1j * np.matmul(state, self.b_i))
        return A

    def unnormalized_probability(self, state):
        return np.abs(self.unnormalized_amplitude(state))**2

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

    @profile
    def ising_local_energy(self, states: np.ndarray):
        gamma = self.hamiltonian
        p_i = self.unnormalized_amplitude(states)

        def _create_mu(state, index):
            mu = state.copy()
            mu[:, index] = 1 - state[:, index]

            if index == self.visible_size-1:
                pass
            else:
                mu[:, index+1] = 1 - state[:, index+1]
            return mu

        def _local_index_energy(gamma_values, index):
            mu_i = _create_mu(states, index)

            p_j = self.unnormalized_amplitude(mu_i)

            return gamma_values * p_j / p_i

        local_energy = sum([_local_index_energy(g, j) for (j, g) in enumerate(gamma)])

        return np.asarray(local_energy)

    def exact_energy(self):
        wave_function = self.wave_function()
        E = wave_function.conj().T @ self.hamiltonian @ wave_function
        return E.real

    def estimate_energy(self):
        walker = Walker(self.visible_size, self.walker_steps)
        sampled_states = walker(self.probability, self.walker_steps)

        if type(self.hamiltonian) is IsingHamiltonianReduced:
            return np.mean(self.ising_local_energy(sampled_states)).real
        else:
            return np.mean(self.local_energy(sampled_states)).real

    def omega_estimate(self, states):
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

    def finite_grad(self):

        if self.walker_steps == 0:
            func = self.exact_energy
            h = 10e-4
        else:
            func = self.estimate_energy
            h = 3 / np.sqrt(self.walker_steps)

        grad_list = []
        for param in self.params:
            grad_array = np.zeros(param.shape)
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param[i, j] += h
                    E1 = func()
                    param[i, j] -= 2 * h
                    E2 = func()
                    param[i, j] += h
                    grad = (E1 - E2) / (2 * h)
                    grad_array[i, j] = grad

            grad_list.append(grad_array)

        return grad_list

    def analytical_grad(self):

        if self.walker_steps == 0:
            return self.exact_distribution_grad()
        else:
            return self.estimate_distribution_grad()

    def exact_distribution_grad(self):
        grad_list = []
        H = self.hamiltonian
        omega = self.omega(self.all_states)
        wf = self.wave_function()

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

    def estimate_distribution_grad(self):
        walker = Walker(self.visible_size, self.walker_steps)
        states = walker(self.unnormalized_probability, self.walker_steps)
        omega_list = self.omega_estimate(states)

        if type(self.hamiltonian) is IsingHamiltonianReduced:
            local_energies = self.ising_local_energy(states)
        else:
            local_energies = self.local_energy(states)

        grad_list = []

        for omega in omega_list:
            omega_bar = np.mean(omega, axis=0)
            grad = np.mean(np.conj(local_energies)*(omega - omega_bar), axis=0).real * 2
            grad_list.append(grad)

        return grad_list

    @utils.timing
    def train(self,
              iterations=1000,
              lr=0.01,
              analytical_grad=True,
              print_energy=True,
              termination_condition:(tuple) = None):
        energy_list = []

        try:
            for i in range(iterations):
                if analytical_grad:
                    grad_list = self.analytical_grad()
                else:
                    grad_list = self.finite_grad()

                grad_list = self.adam.step(grad_list)

                for param, grad in zip(self.params, grad_list):
                    grad = grad.reshape(param.shape)
                    param -= lr * grad

                if self.walker_steps == 0:
                    energy_list.append(self.exact_energy()[0, 0])
                else:
                    energy_list.append(self.estimate_energy())

                if print_energy:
                    print(f"Current ground state: {energy_list[-1]} for training step {i}")

                if termination_condition:
                    if utils.relative_error(energy_list[-1], termination_condition[1]) < termination_condition[0]:
                        break

        except KeyboardInterrupt:
            print(f"Training interrupted by user")

        return energy_list


class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, grad_list):
        self.t += 1
        if self.t == 1:
            self.m = [np.zeros_like(grad) for grad in grad_list]
            self.v = [np.zeros_like(grad) for grad in grad_list]

        mod_grad_list = []
        for i, grad in enumerate(grad_list):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            mod_grad_list.append(m_hat / (np.sqrt(v_hat) + self.eps))

        return mod_grad_list


class Walker(object):

    def __init__(self,
                 visible_size: int,
                 steps: int,
                 ):

        self.steps = steps
        self.burn_in = self.steps // 10
        self.current_state = np.random.randint(0, 2, visible_size)
        self.next_state = copy.deepcopy(self.current_state)
        self.walk_results = []

    def __call__(self, function, num_steps):

        self.estimate_distribution(function)
        return self.get_history()

    def get_history(self):
        return np.asarray(self.walk_results)

    def clear_history(self):
        self.walk_results = []

    def estimate_distribution(self, function, burn_in=True) -> None:
        self.clear_history()

        if burn_in:
            self.burn_in_walk(function)

        self.random_walk(function)

    def random_walk(self, function):

        for i in range(self.steps):
            self.next_state = utils.hamming_steps(self.current_state)
            self.walk_results.append(self.current_state)

            if self.acceptance_criterion(function):
                self.current_state = copy.deepcopy(self.next_state)

            else:
                self.next_state = copy.deepcopy(self.current_state)

    def burn_in_walk(self, function):
        for i in range(self.burn_in):
            self.next_state = utils.hamming_steps(self.current_state)

            if self.acceptance_criterion(function):
                self.current_state = copy.deepcopy(self.next_state)
            else:
                self.next_state = copy.deepcopy(self.current_state)

    def acceptance_criterion(self, function) -> bool:
        u = np.random.uniform(0, 1)
        new_score = function(self.next_state)
        old_score = function(self.current_state)

        score = new_score / old_score > u

        return score
