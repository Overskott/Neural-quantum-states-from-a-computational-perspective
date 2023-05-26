from matplotlib import pyplot as plt

from config_parser import get_config_file
from src import nqs
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
# ex_energy_list = [rbm.train_mcmc(iterations=100, lr=0.01, print_energy=True)]
#
#
# for ex in ex_energy_list:
#     plt.plot(ex)
#     plt.xlabel('Gradient steps')
#     plt.ylabel('Energy')
#
# plt.show()

n = 4
hidden = 8
steps = 1000
walker_steps = 0
np.set_printoptions(linewidth=200, precision=4, suppress=True)
np.random.seed(42)

gamma = random_gamma(n)
#H = nqs.IsingHamiltonianReduced(gamma=gamma)
H_check = H = nqs.RandomHamiltonian(n=n)

eig, state = np.linalg.eigh(H_check)
print(f"Eig: {eig},state: {state}")
E_truth = np.min(eig)
e_truth_index = np.where(eig == E_truth)[0]
print(f"e_truth_index: {e_truth_index}")
print(f"E_truth: {E_truth}, state truth: {state[e_truth_index]}")
plt.axhline(y=E_truth, color='b', linestyle='--')

rbm = nqs.RBM(visible_size=n, hidden_size=hidden, hamiltonian=H, walker_steps=walker_steps)
energy_list = [it for it in rbm.train(iterations=steps, lr=0.01, print_energy=True)]

print(f"RBM energy: {energy_list[-1]}, Rbm state: {rbm.wave_function()}")

print(f"Energy error: {np.abs(energy_list[-1]-E_truth)}")
print(f"State error {state[e_truth_index] @ rbm.wave_function()}")
#print(rbm.train.run_time)

