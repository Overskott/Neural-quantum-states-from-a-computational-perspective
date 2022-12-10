from src.mcmc import *
from src.ansatz import RBM
from src.utils import *
from src.Model import Model
from config_parser import get_config_file
from scipy import optimize
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
    H = random_hamiltonian(2**visible_layer_size)  # Hamiltonian

    walker = Walker()
    rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters
    model = Model(rbm, walker, H)  # Initializing model with RBM and Hamiltonian

    model.walker.estimate_distribution(model.rbm.probability)  # Estimate the distribution
    history = [utils.binary_array_to_int(state) for state in model.walker.get_history()]

    # Printing results
    print(f"Accept rate: {model.walker.average_acceptance()}")
    # print(f"Data: {walker.get_walk_results()}" )
    # Plotting histogram with results

    result_list = []

    #rbm = RBM(start_state, visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters

    for i in range(2 ** visible_layer_size):
        result_list.append(model.rbm.probability(int_to_binary_array(i, visible_layer_size)))

    # Plotting histogram with results
    norm = sum(result_list)


    print(f"Expectation energy:  {min(sum(result_list)/len(result_list))}")
    print(f"Estimated energy: {model.estimate_energy()}")
    print(f"Exact energy: {np.linalg.eigvalsh(H)[0]}")



    plt.figure(0)
    plt.hist(history, density=True, bins=range(2**visible_layer_size+1), edgecolor="black", align='left', rwidth = 0.8)
    plt.scatter([x for x in range(2**visible_layer_size)], (result_list / norm), color='red')
    plt.title("RBM Probability Distribution")
    plt.xlabel('State')
    plt.ylabel('Probalility')
    plt.legend(['Analytic Results', 'MCMC Results'])

    plt.show()







