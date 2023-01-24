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

    #H =np.array([[-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    H = np.diag([-2, 0, 2, 4])  # Hamiltonian

    walker = Walker()
    rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters
    model = Model(rbm, walker, H)  # Initializing model with RBM and Hamiltonian

    model.walker.estimate_distribution(model.rbm.probability)  # Estimate the distribution


    # Printing results
    print(f"Accept rate: {model.walker.average_acceptance()}")
    # print(f"Data: {walker.get_walk_results()}" )
    # Plotting histogram with results



    # Plotting histogram with results

    print(f"Estimated energy: {model.estimate_energy()}")
    print(f"Exact energy: {np.linalg.eigvalsh(H)}")

    model.gradient_descent_2()

    # seed = 44  # Seed for random number generator
    # np.random.seed(seed)
    #
    # b = random_complex_array(visible_layer_size)  # Visible layer bias
    # c = random_complex_array(hidden_layer_size)  # Hidden layer bias
    # W = random_complex_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights
    # # H = random_hamiltonian(2**visible_layer_size)  # Hamiltonian
    #
    # H = np.array([[-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    # # H = np.diag([-3, 1, -1, 1, 2, 3, -4, 5])  # Hamiltonian
    #
    # walker = Walker()
    # rbm = RBM(visible_bias=b, hidden_bias=c,
    #           weights=W)  # Initializing RBM currently with random configuration and parameters
    # model = Model(rbm, walker, H)  # Initializing model with RBM and Hamiltonian
    #
    # model.walker.estimate_distribution(model.rbm.probability)  # Estimate the distribution
    #
    # model.gradient_descent_1(model.rbm.get_parameters_as_array())

    print(f"Estimated energy: {model.estimate_energy()}")

    states_list = [int_to_binary_array(i, visible_layer_size) for i in range(2 ** visible_layer_size)]
    result_list = np.asarray([model.rbm.probability(state) for state in states_list])
    norm = sum(result_list)

    model.walker.estimate_distribution(model.rbm.probability)
    history = [utils.binary_array_to_int(state) for state in model.walker.get_history()]

    #print(f"Expectation energy:  {model.estimate_energy(states_list)}")
    plt.figure(0)
    plt.hist(history, density=True, bins=range(2**visible_layer_size+1), edgecolor="black", align='left', rwidth = 0.8)
    plt.scatter([x for x in range(2**visible_layer_size)], (result_list / norm), color='red')
    plt.title("RBM Probability Distribution")
    plt.xlabel('State')
    plt.ylabel('Probalility')
    plt.legend(['Analytic Results', 'MCMC Results'])

    # plt.figure(1)
    # plt.plot([binary_array_to_int(i) for i in states_list], result_list)
    plt.show()







