
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
walker_steps = 1000

parameters = get_config_file()['parameters']

visible_layer_size = parameters['visible_size']  # Number of qubits
hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes

n = visible_layer_size
d = 2**n
np.random.seed(42)
hamiltonian = random_hamiltonian(d)
eig,_ = np.linalg.eigh(hamiltonian)
E_truth = np.min(eig)
print(f"Energy truth: {E_truth}")

rbm = RBM(visible_size=n, hidden_size=hidden_layer_size, hamiltonian=hamiltonian)

print(rbm.exact_energy())
print(rbm.estimate_energy())

ex_energy_list = [rbm.train(iter=1000, lr=0.01)]


for ex in ex_energy_list:
    plt.plot(ex)
    plt.xlabel('Gradient steps')
    plt.ylabel('Energy')

plt.show()

