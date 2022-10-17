import numpy as np


def random_array(size, low=-1, high=1):
    return np.random.uniform(low, high, size)


def random_matrix(size, low=-1, high=1):
    return np.random.uniform(low, high, (size, size))


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


def generate_positive_energy_hamiltonian(n_qubits: int):
    G = np.random.normal(0, 1, (n_qubits, n_qubits))

    H = G + np.transpose(G)

    eig, eigvec = np.linalg.eig(H)
    gs_index = np.argmin(eig)
    gs = eigvec[gs_index]
    gs = gs/(np.sum(gs**2))

    beta = 0
    ground_state = np.min(eig)
    hamiltonian = None

    while ground_state < 0:
        beta += 1

        hamiltonian = H - beta * (np.transpose(gs) @ gs)
        ground_state = np.min(np.linalg.eigvals(hamiltonian))

        if beta % 1000 == 0:
            print(f"Beta: {beta}, ground state: {ground_state}")

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
