import numpy as np

from src.mcmc import *
from src.ansatz import RBM
from src.utils import *
from src.model import Model, Adam
from config_parser import get_config_file

import matplotlib.pyplot as plt

if __name__ == '__main__':

    parameters = get_config_file()['parameters']

    visible_layer_size = parameters['visible_size']  # Number of qubits
    hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes

    seed = 44  # Seed for random number generator
    np.random.seed(seed)

    b = random_complex_array(visible_layer_size)  # Visible layer bias
    c = random_complex_array(hidden_layer_size)  # Hidden layer bias
    W = random_complex_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights
    #H = random_hamiltonian(2**visible_layer_size)  # Hamiltonian
    H = np.diag([-2, 0, -0, 2])  # Hamiltonian
    #H = np.diag([-2, 0, 1, -8, 0, 0, -5, -2])  # Hamiltonian

    walker = Walker()
    rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters
    model = Model(rbm, walker, H)  # Initializing model with RBM and Hamiltonian
    model_copy = copy.deepcopy(model)

    # Printing results
    print(f"Accept rate: {model.walker.average_acceptance()}")

    print(f"Estimated energy: {model.estimate_energy()}")
    print(f"Exact energy: {np.linalg.eigvalsh(H)}")

    fd_plot_list = model.gradient_descent('finite_difference')

    analytic_plot_list = model_copy.gradient_descent('analytical')

    plt.plot(fd_plot_list, label='Finite Difference')
    plt.plot(analytic_plot_list, label='Analytical')
    plt.axhline(y=min(np.linalg.eigvalsh(H)), color='red', linestyle='--', label='Ground State')
    plt.legend()
    plt.show()

    print(f"Estimated energy FD: {model.estimate_energy()}")
    print(f"Estimated energy Analytic: {model_copy.estimate_energy()}")

    model.walker.estimate_distribution(model.rbm.probability)
    history = [utils.binary_array_to_int(state) for state in model.walker.get_history()]

    plt.figure(0)
    plt.hist(history, density=True, bins=range(2**visible_layer_size+1), edgecolor="black", align='left', rwidth = 0.8)
    plt.scatter([x for x in range(2**visible_layer_size)], model.get_prob_distribution(), color='red')
    plt.title("RBM Probability Distribution")
    plt.xlabel('State')
    plt.ylabel('Probalility')
    plt.legend(['Analytic Results', 'MCMC Results'])

    plt.show()







