from matplotlib import pyplot as plt

from config_parser import get_config_file
from src.nqs import RBM, RandomHamiltonian
from src.utils import *

n = 2
hidden = 2


steps = 100

walker_steps = 100

np.set_printoptions(linewidth=200, precision=4, suppress=True)
np.random.seed(42)


H = RandomHamiltonian(n)
print(H)

rbm = RBM(visible_size=n, hidden_size=hidden, hamiltonian=H, walker_steps=walker_steps)

print([binary_array_to_int(state) for state in rbm.mcmc_dist()])
