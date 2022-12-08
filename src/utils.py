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


def random_hamiltonian(n_qubits: int):
    """Generate a random hamiltonian matrix of size n_qubits x n_qubits"""
    re = np.random.normal(0, 1, (n_qubits, n_qubits))
    im = np.random.normal(0, 1, (n_qubits, n_qubits)) * 1j
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

    bit_array = np.zeros((length, 1), dtype='i4')

    for i, c in enumerate(binary[::-1]):
        bit_array[i] = int(c)

    return np.flip(bit_array)  # Flipping (reversing) to return in 'least significant bit' format


def binary_array_to_int(binary_array):
    """Updated the self.value value based on the bit_array value"""
    value = 0

    for i, bit in enumerate(np.flip(binary_array)):
        value += bit * 2 ** i

    return value


def finite_difference(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)


def get_parameter_derivative(params: List[float], func, h=1e-5):

    params_deriv = []

    for param in params:
        params_deriv.append(finite_difference(func, param, h))

    return params_deriv


def gradient_descesnt(func, params, learning_rate=0.01, n_steps=1000):
    params = np.array(params)
    for i in range(n_steps):
        params = params - learning_rate * np.array(get_parameter_derivative(params, func))
    return params