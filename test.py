
import timeit

import numpy as np


from config_parser import get_config_file
from src import utils
from src.ansatz import RBM
from src.hamiltionians import Hamiltonian, IsingHamiltonian, DiagonalHamiltonian
from src.mcmc import Walker
from src.model import Model
from src.utils import *


seed = 44  # Seed for random number generator
#np.random.seed(seed)

parameters = get_config_file()['parameters']

visible_layer_size = parameters['visible_size']  # Number of qubits
hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes

b = random_complex_array(visible_layer_size)  # Visible layer bias
c = random_complex_array(hidden_layer_size)  # Hidden layer bias
W = random_complex_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights

gamma = random_gamma(visible_layer_size)

walker = Walker()
rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters
model = Model(rbm, walker, gamma)  # Initializing model with RBM and Hamiltonian

h = Hamiltonian(2**visible_layer_size)

print(h)
print(type(h))

ih = IsingHamiltonian(2**visible_layer_size)
dh = DiagonalHamiltonian(2**visible_layer_size, 2)
print(type(ih) == IsingHamiltonian)
print(dh)