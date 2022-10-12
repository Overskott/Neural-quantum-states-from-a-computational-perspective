
import random
import numpy as np


def random_hamiltonian(n_qubits: int, low=-1, high=1):
    """Generate a random hamiltonian matrix of size n_qubits x n_qubits"""
    re = np.random.uniform(low, high, (n_qubits, n_qubits))
    im = np.random.uniform(low, high, (n_qubits, n_qubits)) * 1j
    ginibre = re + im

    hamiltonian = ginibre + ginibre.T.conj()

    return hamiltonian


class Hamiltonian():

    def __init__(self, size):
        self.size = size
        self.h_max = 2
        self.h = [random.uniform(-1, 1)*self.h_max for _ in range(self.size)]


