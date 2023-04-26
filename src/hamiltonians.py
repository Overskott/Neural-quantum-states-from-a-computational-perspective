
import numpy as np
import src.utils as utils


class Hamiltonian(np.ndarray):

    def __new__(cls, n):
        h = utils.random_hamiltonian(2 ** n)
        obj = np.asarray(h).view(cls)
        return obj

    def __array_finalize__(self, obj, **kwargs):
        if obj is None:
            return

    def __call__(self):
        return h

class IsingHamiltonian(Hamiltonian):

    def __new__(cls, n):
        ih = utils.random_ising_hamiltonian(n)
        obj = np.asarray(ih).view(cls)
        return obj


class ReducedIsingHamiltonian(Hamiltonian):

    def __new__(cls, n):
        ih = utils.random_gamma(n)
        obj = np.asarray(ih).view(cls)
        return obj


class DiagonalHamiltonian(Hamiltonian):

    def __new__(cls, n, diagonal=0):
        dh = utils.random_diagonal_hamiltonian(2 ** n, diagonal)
        obj = np.asarray(dh).view(cls)
        return obj

    def __init__(self, n, diagonal=0):
        super().__init__(n)
        self.diagonal = diagonal
