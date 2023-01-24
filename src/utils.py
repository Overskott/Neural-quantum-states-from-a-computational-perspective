from typing import List

import numpy as np


def random_array(size, mu=0, sigma=1):
    return np.random.normal(mu, sigma, size)


def random_complex_array(size, mu=0, sigma=1):
    re = np.random.normal(mu, sigma, size)
    im = np.random.normal(mu, sigma, size) * 1j

    return re + im


def random_matrix(size_x, size_y, mu=0, sigma=1):
    return np.random.normal(mu, sigma, (size_x, size_y))


def random_complex_matrix(size_x, size_y, mu=0, sigma=1):
    re = np.random.normal(mu, sigma, (size_x, size_y))
    im = np.random.normal(mu, sigma, (size_x, size_y)) * 1j

    return re + im


@DeprecationWarning
# use random_hamiltonian instead
def random_symmetric_matrix(size, mu=-1, sigma=1):
    a = np.random.normal(mu, sigma, (size, size))
    return np.tril(a) + np.tril(a, -1).T


def random_hamiltonian(size: int):
    """Generate a random hamiltonian matrix of size n_qubits x n_qubits"""
    re = np.random.normal(0, 1, (size, size))
    im = np.random.normal(0, 1, (size, size)) * 1j
    ginibre = re + im

    hamiltonian = ginibre + ginibre.T.conj()

    return hamiltonian


def generate_positive_ground_state_hamiltonian(n_qubits: int):
    size = 2**n_qubits
    G = np.random.normal(0, 1, (size, size))

    H = G + np.transpose(G)

    a = np.random.uniform(0, 1, size)
    a = a / (np.sum(a ** 2))

    beta = 0
    hamiltonian = 0
    gs = np.array([-1, 1])

    while not (np.all(gs > 0) or np.all(gs < 0)):
        beta += 0.1

        hamiltonian = H - beta * (np.transpose(a) @ a)

        eig, eigvec = np.linalg.eig(hamiltonian)
        gs_index = np.argmin(eig)
        gs = eigvec[:, gs_index]
        gs = gs / (np.sum(gs ** 2))

    return hamiltonian


def int_to_binary_array(value, length):

    binary = format(value, 'b')
    bit_array = np.zeros(length, dtype='i4')

    for i, c in enumerate(binary[::-1]):
        bit_array[i] = int(c)

    return np.flip(bit_array)  # Flipping (reversing) to return in 'least significant bit' format


def binary_array_to_int(binary_array):
    """Updated the self.value value based on the bit_array value"""
    value = 0

    for i, bit in enumerate(np.flip(binary_array)):
        value += bit * 2 ** i

    return value


def flip_bit(state: np.ndarray , index: int):
    """Flips (0->1 or 1->0) the bit on given index of the state"""
    state[index] = 1 - state[index]


def normal_distribution(x) -> float:
    """

    :param x:
    :param sigma:
    :param mu:
    """
    sigma = 1
    mu = 0

    if type(x) is not int:
        x = binary_array_to_int(x)

    _1 = 1 / (sigma * np.sqrt(2 * np.pi))
    _2 = -(1 / 2) * ((x - mu) / sigma) ** 2

    return _1 * np.exp(_2)


def double_normal_distribution(x: int, distance: int, sigma_1: float, mu_1: float, sigma_2: float, mu_2: float):
    return (normal_distribution(x, sigma_1, mu_1) + normal_distribution(x + distance, sigma_2, mu_2)) / 2