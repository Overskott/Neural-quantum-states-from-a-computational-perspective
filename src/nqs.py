import line_profiler
profile = line_profiler.LineProfiler()

import numpy as np
from tqdm import tqdm

import src.utils as utils
from config_parser import get_config_file
from src.mcmc import Walker
from src.optimization import Adam


class Hamiltonian(np.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, dim=None, values=None):

        self._check_dim(dim)

        if dim is None:
            self.dim = np.asarray(values).shape[0]
        else:
            self.dim = 2**dim

        if values is None:
            self.values = np.eye(self.dim)
        else:
            self.values = np.asarray(values)

        #self._check_values(self.values)

    def __repr__(self):
        return f"{self.__class__.__name__}\n({self.values})"

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __array__(self):
        return self.values

    def _check_dim(self, dim):
        if dim is None:
            pass
        elif type(dim) != int:
            raise TypeError(f"dim must be positive int, not {type(dim)}")
        elif dim <= 0:
            raise ValueError(f"dim must be positive int, not {dim}")

    def _check_values(self, values):
        if values is None:
            pass
        elif values.shape != (self.dim, self.dim):
            raise ValueError(f"values must be of shape ({self.dim}, {self.dim}), not {values.shape}")


class RandomHamiltonian(Hamiltonian):

    def __init__(self, dim):
        super().__init__(dim, values=None)

        self.values = utils.random_hamiltonian(2**dim)


class IsingHamiltonian(Hamiltonian):

    def __init__(self, dim, values=None, gamma=None):
        super().__init__(dim, values)

        self.values = utils.random_ising_hamiltonian(dim, gamma)


class IsingHamiltonianReduced(Hamiltonian):

    def __init__(self, dim=None, values=None):
        super().__init__(dim, values)

        self.values = utils.random_gamma(dim)


class RBM(object):

    def __init__(self,
                 visible_size: np.ndarray = None,
                 hidden_size: np.ndarray = None,
                 hamiltonian: Hamiltonian = None,
                 walker_steps: int = None
                 ):

        data = get_config_file()['parameters']  # Load the config file
        scale = 1#/np.sqrt(n_hid)

        if visible_size is None:
            self.visible_size = data['visible_size']
        else:
            self.visible_size = visible_size

        if hidden_size is None:
            self.hidden_size = data['hidden_size']
        else:
            self.hidden_size = hidden_size

        if hamiltonian is None:
            self.hamiltonian = utils.random_hamiltonian(2**self.visible_size)
        else:
            self.hamiltonian = hamiltonian

        if walker_steps is None:
            self.walker_steps = data['walker_steps']
        else:
            self.walker_steps = walker_steps

        self.adam = Adam()

        self.b_r = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias #
        self.b_i = np.random.normal(0, 1/scale, (self.visible_size, 1))  # Visible layer bias #

        self.c_r = np.random.normal(0, 1/scale, (1, self.hidden_size))  # Hidden layer bias
        self.c_i = np.random.normal(0, 1/scale, (1, self.hidden_size))  # Hidden layer bias

        self.W_r = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights
        self.W_i = np.random.normal(0, 1/scale, (self.visible_size, self.hidden_size))  # s - h weights

        self.params = [self.b_r, self.b_i, self.c_r, self.c_i, self.W_r, self.W_i]

        # Generate all possible states
        self.all_states = self.get_all_states()

    def get_all_states(self):
        all_states_list = []

        for i in range(2 ** self.visible_size):
            state = utils.numberToBase(i, 2, self.visible_size)
            all_states_list.append(state)

        return np.array(all_states_list)

    def wave_function(self):
        return self.normalized_amplitude(self.all_states)

    def probability_dist(self):
        return np.abs(self.wave_function())**2

    def normalized_amplitude(self, state):
        # Normalized amplitude_old
        Z = np.sqrt(np.sum(np.abs(self.unnormalized_amplitude(self.all_states)) ** 2))
        return self.unnormalized_amplitude(state) / Z

    def probability(self, state: np.ndarray) -> float:
        """ Calculates the probability of finding the RBM in state s """
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

        def create_mu(state, index):
            mu = state.copy()
            mu[:, index] = 1 - state[:, index]

            # if index == self.visible_size-1:
            #     pass
            # else:
            mu[:, index+1] = 1 - state[:, index+1]
            return mu

        def eloc_index_value(gamma, index):
            mu_i = create_mu(states, index)
            p_j = self.unnormalized_amplitude(mu_i)
            return gamma * p_j / p_i

        local_energy = sum([eloc_index_value(g, j) for (j, g) in enumerate(gamma)])

        return np.asarray(local_energy)

    def exact_energy(self):
        wave_function = self.wave_function()
        E = wave_function.conj().T @ self.hamiltonian @ wave_function
        return E.real

    def estimate_energy(self):
        walker = Walker()
        sampled_states = walker(self.probability, self.walker_steps)

        if self.hamiltonian is IsingHamiltonianReduced:
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

    def finite_grad(self, h=0.05):
        if self.walker_steps == 0:
            func = self.exact_energy
        else:
            func = self.estimate_energy

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
        walker = Walker()
        states = walker(self.unnormalized_probability, self.walker_steps)
        omega_list = self.omega_estimate(states)
        local_energies = self.local_energy(states)

        grad_list = []

        for omega in omega_list:
            omega_bar = np.mean(omega, axis=0)
            grad = np.mean(np.conj(local_energies)*(omega - omega_bar), axis=0).real * 2
            grad_list.append(grad)

        return grad_list


    def train(self, iter, lr=0.01, analytical_grad=False, print_energy=False):
        energy_list = []

        try:
            for i in range(iter):
                if analytical_grad:
                    grad_list = self.analytical_grad()
                else:
                    grad_list = self.finite_grad()

                grad_list = self.adam.step(grad_list)

                for param, grad in zip(self.params, grad_list):
                    #print(f"params: {param}, grad:  {grad}")
                    #print(f"params shape: {param.shape}, grad shape:  {grad.shape}")
                    param -= lr * grad

                energy_list.append(self.exact_energy()[0, 0])

                if print_energy:
                    print(f"Current ground state: {energy_list[-1]} for training step {i}")

        except KeyboardInterrupt:
            print(f"Training interrupted by user")

        return energy_list

    @utils.timing
    def train_mcmc(self, iter=1000, lr=0.01, analytical_grad=True, print_energy=False, termination_condition=None):
        energy_list = []

        try:
            for i in range(iter):
                if analytical_grad:
                    grad_list = self.analytical_estimate_grad()
                else:
                    grad_list = self.finite_grad()
                grad_list = self.adam.step(grad_list)
                for param, grad in zip(self.params, grad_list):
                    grad = grad.reshape(param.shape)
                    #print(f"params: {param}, grad:  {grad}")
                    #print(f"params shape: {param.shape}, grad shape:  {grad.shape}")

                    param -= lr * grad
                energy_list.append(self.estimate_energy())
                if print_energy:
                    print(f"Current ground state: {energy_list[-1]} for training step {i}")
                #input()

        except KeyboardInterrupt:
            print(f"Training interrupted by user")

        return energy_list
