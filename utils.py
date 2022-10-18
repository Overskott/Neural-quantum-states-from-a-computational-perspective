import numpy as np


def random_array(size, low=-1, high=1):
    return np.random.uniform(low, high, size)


def random_matrix(size_x, size_y, low=-1, high=1):
    return np.random.uniform(low, high, (size_x, size_y))


@DeprecationWarning
def random_symmetric_matrix(size, low=-1, high=1):
    a = np.random.uniform(low, high, (size, size))
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
