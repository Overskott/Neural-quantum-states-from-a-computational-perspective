from matplotlib import pyplot as plt

from config_parser import get_config_file
from src.nqs import RBM, RandomHamiltonian, Hamiltonian, IsingHamiltonian, IsingHamiltonianReduced
from src.utils import *

n = 2

hidden = 2


steps = 100

walker_steps = 100
np.random.seed(42)



A = random_hamiltonian(2)
print(A.shape[0])
print(A.shape[1])
H = Hamiltonian()

rh = RandomHamiltonian(n)
print(rh)

ih = IsingHamiltonian(n)
print(ih)
ihr = IsingHamiltonianReduced(n)
print(ihr)

gamma = random_gamma(n)

ih = IsingHamiltonian(gamma=gamma)
print(ih)
ihr = IsingHamiltonianReduced(gamma=gamma)
print(ihr)

rbm = rbm = RBM(visible_size=2, hidden_size=2, hamiltonian=rh, walker_steps=0)

print(rbm.params)
