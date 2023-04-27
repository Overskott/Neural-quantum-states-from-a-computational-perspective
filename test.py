
import timeit

import numpy as np
from matplotlib import pyplot as plt

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

n = 4
d = 2**n
np.random.seed(42)
hamiltonian = random_hamiltonian(d)
eig,_ = np.linalg.eigh(hamiltonian)
E_truth = np.min(eig)
print(f"Energy truth: {E_truth}")

rbm = RBM(visible_size=n, hidden_size=8, hamiltonian=hamiltonian)
energy_list = rbm.train(iter=500, lr=0.01, analytical_grad=True)

plt.plot(energy_list)
plt.show()