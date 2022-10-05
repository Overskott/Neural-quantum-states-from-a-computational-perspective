import numpy as np


def random_array(size, low=-1, high=1):
    return np.random.uniform(low, high, size)


def random_matrix(size, low=-1, high=1):
    return np.random.uniform(low, high, (size, size))


def random_symmetric_matrix(size, low=-1, high=1):
    a = np.random.uniform(low, high, (size, size))
    return np.tril(a) + np.tril(a, -1).T


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

