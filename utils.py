import numpy as np


def random_array(size, low=-1, high=1):
    return np.random.uniform(low, high, size)


def random_matrix(size, low=-1, high=1):
    return np.random.uniform(low, high, (size, size))


def random_symmetric_matrix(size, low=-1, high=1):
    a = np.random.uniform(low, high, (size, size))
    return np.tril(a) + np.tril(a, -1).T


def random_hamiltonian(n_qubits: int, low=-1, high=1):
    """Generate a random hamiltonian matrix of size n_qubits x n_qubits"""
    re = np.random.uniform(low, high, (n_qubits, n_qubits))
    im = np.random.uniform(low, high, (n_qubits, n_qubits)) * 1j
    ginibre = re + im

    hamiltonian = ginibre + ginibre.T.conj()

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


def minimize_rbm_energy(rbm, x_0):
    pass


def create_variable_array(rbm):
    """Creates a variable array from the RBM variables"""
    return np.concatenate((rbm.b, rbm.c, rbm.W.flatten()))  # Flattening the weights matrix

def set_rbm_variables(rbm, x_0: np.ndarray):
    """
    Sets the RBM variables to the values in x_0

    b = x_0[:len(rbm.b)] is the visible layer bias
    c = x_0[len(rbm.b):len(rbm.c)] is the hidden layer bias
    W = x_0[np.shape(W)] is the weights

    """

    pass
    #rbm.b =
    #rbm.c =
    #rbm.W =

    #return rbm