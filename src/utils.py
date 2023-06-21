import numpy as np
import random


def random_hamiltonian(n: int):
    """
    Generate and returns a random hamiltonian matrix with dimensions n^2 x n^2.

    :param n: The number of qubits in the system of the Hamiltonian matrix.
    :return H: Hamiltonian matrix with random elements.
    """
    H = np.random.normal(0, 1, (2**n, 2**n)) + 1j * np.random.normal(0, 1, (2**n, 2**n))
    H = H + np.conj(H).T
    return H


def random_gamma(n: int, sigma=0, mu=1) -> np.ndarray:
    """
    Generate a random gamma array of size n-1 with normal distribution. Used for generating
    random IsingHamiltonian and random ReducedIsingHamiltonian.

    :param n: Number of qubits in the system.
    :param sigma: The standard deviation of the normal distribution.
    :param mu: The mean of the normal distribution.

    :return: np.ndarray of size n-1 with random gamma values.
    """
    return np.random.normal(size=n-1, loc=sigma, scale=mu)


def random_ising_hamiltonian(n: int = None, gamma_array: np.ndarray = None):
    """
    Generate a random Ising Hamiltonian matrix of size n^2 x n^2 with random gamma values. Only provide one
    of the parameters n or gamma_array.
    :param n: Number of qubits in the system.
    :param gamma_array: The gamma values to use for the Ising Hamiltonian.

    :return: The Ising Hamiltonian matrix.
    """
    if gamma_array is None:
        n = n
        gamma = np.random.normal(0, 1, n - 1)
    else:
        n = len(gamma_array) + 1
        gamma = gamma_array

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


def timing(f):
    """
    Decorator for timing functions based on the following example:
    https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator.

    Also adds a run_time attribute to the function decorated. run_time can be
    accessed as f.run_time.

    :param f: The function to time
    :return:
    """

    from functools import wraps
    from time import time

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        wrap.run_time = te - ts # Add the run_time attribute to the function decorated.
        print(f"func:{f.__name__} args:[{args}, {kw}] took: {te-ts} sec")
        return result
    return wrap


def binary_array_to_int(binary_array: np.ndarray) -> int:
    return int(''.join(map(lambda x: str(int(x)), binary_array)), 2)


def numberToBase(n: int, b: int, num_digits: int) -> list[int]:
    """
    Convert a number to a given base with a given number of digits and return a list with the digits.

    :param n: The number to convert
    :param b: The base to convert to (e.g. base 10). Max base is 10.
    :param num_digits: The number of digits to use in the conversion

    :return: The list of digits in the given base
    """
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits) < num_digits:
        digits.append(0)
    return digits[::-1]


def flip_bit(state: np.ndarray, index: int):
    """
    Flips (0->1 or 1->0) the bit on given index of the state.

    :param state: The state to flip the bit in
    :param index: The index of the bit to flip

    :return: The state with the flipped bit
    """
    state[index] = 1 - state[index]


def hamming_steps(binary_array: np.ndarray, flips: int = 1) -> np.ndarray:
    """
    Perform a number of hamming steps on a binary array.

    :param binary_array: The binary array to perform the hamming steps on
    :param flips: The number of flips to perform

    :return: The binary array after the hamming steps
    """

    new_array = binary_array.copy()
    used_indexes = []
    for i in range(flips):
        flip_index = random.randint(0, binary_array.size-1) # minus 1?

        while flip_index in used_indexes:
            flip_index = random.randint(0, binary_array.size-1)

        used_indexes.append(flip_index)

        new_array[flip_index] = 1 - binary_array[flip_index]

    return new_array


def hamming_step(binary_array: np.ndarray) -> np.ndarray:
    """
    Perform a single hamming step on a binary array.

    :param binary_array: The binary array to perform the hamming step on
    :return: The binary array after the hamming step
    """
    new_array = binary_array.copy()
    flip_index = random.randint(0, binary_array.size-1) # minus 1?
    new_array[flip_index] = 1 - binary_array[flip_index]

    return new_array


@DeprecationWarning
def random_diagonal_hamiltonian(size: int, off_diagonal=0):
    """
    Generate a random diagonal hamiltonian matrix of n n_qubits x n_qubits with off_diagonal elements.

    :param size: Size of the hamiltonian matrix
    :param off_diagonal: Number of off-diagonals above and below the main diagonal

    :return: Diagonal hamiltonian matrix
    """
    H = random_hamiltonian(2**size)
    diag_ham = -(H - np.triu(H, -off_diagonal) - np.tril(H, off_diagonal))

    return diag_ham

@DeprecationWarning
def get_matrix_off_diag_range(H):
    hamiltonian_size = H.shape[0]

    for i in range(hamiltonian_size):

        off_diag = H - np.tril(H, i) + H - np.triu(H, -i)

        if np.count_nonzero(off_diag) == 0:
            return i


@DeprecationWarning
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

