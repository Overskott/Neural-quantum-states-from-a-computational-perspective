
import numpy as np
import src.utils as utils


class Hamiltonian(np.ndarray):

    def __new__(cls, size):
        h = utils.random_hamiltonian(size)
        obj = np.asarray(h).view(cls)
        return obj

    def __array_finalize__(self, obj, **kwargs):
        if obj is None:
            return


class IsingHamiltonian(Hamiltonian):

    def __new__(cls, size):
        ih = utils.random_ising_hamiltonian(size)
        obj = np.asarray(ih).view(cls)
        return obj


class DiagonalHamiltonian(Hamiltonian):

    def __new__(cls, size, diagonal=0):
        dh = utils.random_diagonal_hamiltonian(size, diagonal)
        obj = np.asarray(dh).view(cls)
        return obj
