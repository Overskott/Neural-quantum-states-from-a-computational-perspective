import copy
import timeit

import numpy as np
from matplotlib import pyplot as plt

from config_parser import get_config_file
#from src import utils, nqs
from src.ansatz import RBM
from src.hamiltonians import Hamiltonian, IsingHamiltonian, ReducedIsingHamiltonian, DiagonalHamiltonian
from src.mcmc import Walker
from src.model import NQS
from src.utils import *


import timeit

import numpy as np
from matplotlib import pyplot as plt

from config_parser import get_config_file
from src import utils, nqs
from src.ansatz import RBM
from src.hamiltonians import Hamiltonian, IsingHamiltonian, ReducedIsingHamiltonian, DiagonalHamiltonian
from src.mcmc import Walker
from src.utils import *


seed = 42  # Seed for random number generator
np.random.seed(seed)


parameters = get_config_file()['parameters']

visible_layer_size = parameters['visible_size']  # Number of qubits
hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes
gradient_steps = parameters['gradient_steps']  # Number of gradient descent steps
walker_steps = parameters['walker_steps']  # Number of MCMC steps

n = visible_layer_size
d = 2**n
np.random.seed(42)


# gamma = random_gamma(n-1)
# hamiltonian = random_ising_hamiltonian(n, gamma)
# eig,_ = np.linalg.eigh(hamiltonian)
# E_truth = np.min(eig)
# print(f"Energy truth: {E_truth}")
#
# rbm = RBM(visible_size=n, hidden_size=hidden_layer_size, hamiltonian=gamma, walker_steps=walker_steps)
#
# ex_energy_list = [rbm.train_mcmc(iter=100, lr=0.01, print_energy=True)]
#
#
# for ex in ex_energy_list:
#     plt.plot(ex)
#     plt.xlabel('Gradient steps')
#     plt.ylabel('Energy')
#
# plt.show()



n = 3
hidden = 12
steps = 100
np.set_printoptions(linewidth=200, precision=4, suppress=True)
np.random.seed(42)

H_r = nqs.RandomHamiltonian(n=n)

eig,_ = np.linalg.eigh(H_r)
E_truth = np.min(eig)
print(f"E_truth: {E_truth}")
plt.axhline(y=E_truth, color='b', linestyle='--')

rbm = RBM(visible_size=n, hidden_size=hidden, hamiltonian=H_r)
energy_list = [it for it in rbm.train(iter=steps, lr=0.01, print_energy=True)]


gamma = utils.random_gamma(n)
H_i = nqs.IsingHamiltonian(gamma=gamma)
eig,_ = np.linalg.eigh(H_i)
I_truth = np.min(eig)
print(f"E_truth: {I_truth}")
plt.axhline(y=I_truth, color='k', linestyle='--')

rbm = RBM(visible_size=n, hidden_size=hidden, hamiltonian=H_i)
rbm_ir = copy.deepcopy(rbm)
i_energy_list = [it for it in rbm.train(iter=steps, lr=0.01, print_energy=True)]

ir_energy_list = [it for it in rbm_ir.train(iter=steps, lr=0.01, print_energy=True)]

plt.plot(energy_list)

plt.plot(i_energy_list)

plt.plot(ir_energy_list, 'r--')


plt.legend(['Truth random','Truth Ising', 'Random', 'Ising', 'Ising Reduced' ])

plt.show()
