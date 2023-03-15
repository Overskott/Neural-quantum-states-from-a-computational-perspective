import time
from typing import List

import numpy as np
import random


def random_array(size, mu=0, sigma=1):
    return np.random.normal(mu, sigma, size)


def random_binary_array(size):
    return np.random.randint(0, 2, size)


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
    """Generate a random hamiltonian matrix of n n_qubits x n_qubits"""
    re = np.random.normal(0, 1, (size, size))
    im = np.random.normal(0, 1, (size, size)) * 1j
    ginibre = re + im

    hamiltonian = ginibre + ginibre.T.conj()

    return hamiltonian


def random_gamma(size: int) -> np.ndarray:
    return random_array(size, mu=0, sigma=1)


def random_diagonal_hamiltonian(size: int, off_diagonal=0):
    """
    Generate a random diagonal hamiltonian matrix of n n_qubits x n_qubits with off_diagonal elements.

    :param size: Size of the hamiltonian matrix
    :param off_diagonal: Number of off-diagonals above and below the main diagonal

    :return: Diagonal hamiltonian matrix
    """
    H = random_hamiltonian(size)
    diag_ham = -(H - np.triu(H, -off_diagonal) - np.tril(H, off_diagonal))

    return diag_ham


def get_matrix_off_diag_range(H):
    hamiltonian_size = H.shape[0]

    for i in range(hamiltonian_size):

        off_diag = H - np.tril(H, i) + H - np.triu(H, -i)

        if np.count_nonzero(off_diag) == 0:
            return i


def random_ising_hamiltonian(size: int):

    n = size
    gamma = np.random.normal(0, 1, n - 1)
    # gamma = np.zeros(n-1) - 1
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    XX = np.kron(X, X)

    H = 0
    for i in range(n - 1):
        h = None
        if i != 0:
            h = I

        for j in range(i - 1):
            h = np.kron(h, I)

        if h is None:
            h = XX
        else:
            h = np.kron(h, XX)

        for j in range(i + 2, n):
            h = np.kron(h, I)
        H += gamma[i] * h

    return H

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
    binary_string = format(int(value), 'b').zfill(length)
    binary_array = [int(bit) for bit in binary_string[::-1]]

    return np.flip(np.asarray(binary_array))  # Flipping (reversing) to return in 'least significant bit' format


def binary_array_to_int(binary_array):
    """Updated the self.value value based on the bit_array value"""
    value = sum([bit * 2 ** i for (i, bit) in enumerate(np.flip(binary_array))])

    return int(value)


def flip_bit(state: np.ndarray, index: int):
    """Flips (0->1 or 1->0) the bit on given index of the state"""
    state[index] = 1 - state[index]


def hamming_step(binary_array: np.ndarray) -> np.ndarray:

    new_array = binary_array.copy()
    flip_index = random.randint(0, binary_array.size-1) # minus 1?
    new_array[flip_index] = 1 - binary_array[flip_index]

    return new_array

def hamming_steps(binary_array: np.ndarray, flips: int = 1) -> np.ndarray:

    new_array = binary_array.copy()
    used_indexes = []
    for i in range(flips):
        flip_index = random.randint(0, binary_array.size-1) # minus 1?

        while flip_index in used_indexes:
            flip_index = random.randint(0, binary_array.size-1)

        used_indexes.append(flip_index)

        new_array[flip_index] = 1 - binary_array[flip_index]

    return new_array


def time_function(f, *args, **kwargs):
    start = time.process_time()
    f(*args, **kwargs)
    end = time.process_time()
    return end - start


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