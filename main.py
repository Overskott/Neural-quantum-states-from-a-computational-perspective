from src.mcmc import *
from src.rbm import RBM
from src.utils import *
from config_parser import get_config_file
from scipy import optimize
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parameters = get_config_file()['parameters']

    visible_layer_size = parameters['visible_size']  # Number of qubits
    hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes
    burn_in_steps = parameters['burn_in_steps']  # Number of steps before collecting points
    walker_steps = parameters['walker_steps']  # Number of steps before walker termination
    flips = parameters['hamming_distance']  # Hamming distance traveled between points

    #seed = 42  # Seed for random number generator
    #np.random.seed(seed)

    b = random_complex_array(visible_layer_size)  # Visible layer bias
    c = random_complex_array(hidden_layer_size)  # Hidden layer bias
    W = random_complex_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights
    H = random_hamiltonian(visible_layer_size)  # Hamiltonian

    walker = Walker()
    rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters

    walker.random_walk(rbm.probability, flips)
    history = [state.get_value() for state in walker.get_walk_results()]

    # Printing results
    print(f"Accept rate: {walker.average_acceptance()}")
    # print(f"Data: {walker.get_walk_results()}" )
    # Plotting histogram with results

    result_list = []

    #rbm = RBM(start_state, visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters

    for i in range(2 ** visible_layer_size):
        result_list.append(rbm.probability(int_to_binary_array(i, visible_layer_size)))

    # Plotting histogram with results
    norm = sum(result_list)

    # print(100 * (result_list / norm))

    # for i in range(2 ** bitstring_length):
    #    print(rbm.local_energy(H, walker, i))

    plt.figure(0)
    plt.hist(history, density=True, bins=2**visible_layer_size, edgecolor="black", align='mid')
    plt.scatter([x for x in range(2**visible_layer_size)], (result_list / norm), color='red')
    plt.title("RBM Probability Distribution")
    plt.xlabel('State')
    plt.ylabel('Probalility')
    plt.legend(['Analytic Results', 'MCMC Results'])

    plt.show()







