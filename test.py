
import timeit

import numpy as np


from config_parser import get_config_file
from src import utils
from src.ansatz import RBM
from src.hamiltonians import Hamiltonian, IsingHamiltonian, ReducedIsingHamiltonian, DiagonalHamiltonian
from src.mcmc import Walker
from src.model import Model
from src.utils import *


seed = 44  # Seed for random number generator
np.random.seed(seed)

parameters = get_config_file()['parameters']

visible_layer_size = parameters['visible_size']  # Number of qubits
hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes

b = random_complex_array(visible_layer_size)  # Visible layer bias
c = random_complex_array(hidden_layer_size)  # Hidden layer bias
W = random_complex_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights

H = Hamiltonian(visible_layer_size)

walker = Walker()
rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters
model = Model(rbm, walker, H)  # Initializing model with RBM and Hamiltonian
dist = np.array([[0, 1], [1, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1]])




print(numberToBase(5, 2, 10))
# create matrix with onehot states
# for (row, col) in enumerate(i):
#     print(row, col)
#     M[row, col] = 1
#
#
# J = np.eye(d_2)
#
# local_energy = 0
# p_i = model.rbm.probability(dist)
# for j, int_state in enumerate(M):
#
#     p_j = model.rbm.probability(model.get_all_states())
#     h_ij = int_state @ H @ J
#     print(f"h_ij shape: {h_ij.shape}")
#     print(f"p_i shape: {p_i.shape}")
#     print(f"p_j shape: {p_j.shape}")
#
#     local_energy += sum(h_ij * p_i[j] / p_j)
#
# print(local_energy)