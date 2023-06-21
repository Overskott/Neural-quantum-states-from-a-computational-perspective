from matplotlib import pyplot as plt

from config_parser import get_config_file
from src.nqs import RBM, RandomHamiltonian
from src.utils import *

n = 2
hidden = 2


steps = 100

walker_steps = 100
np.random.seed(42)


H = RandomHamiltonian(n)
print(H)

rbm = RBM(visible_size=n, hidden_size=hidden, hamiltonian=H, walker_steps=walker_steps)

mcmc_dist = [binary_array_to_int(state) for state in rbm.mcmc_dist()]


print(np.unique(mcmc_dist, return_counts=True))


