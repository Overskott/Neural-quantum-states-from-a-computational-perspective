import matplotlib.pyplot as plt
from mcmc import *
from rbm import RBM
from utils import *
from scipy import optimize
from config_parser import get_config_file

if __name__ == '__main__':

    parameters = get_config_file()['parameters']

    visible_layer_size = parameters['visible_size']  # Number of qubits
    hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes
    burn_in_steps = parameters['burn_in_steps']  # Number of steps before collecting points
    walker_steps = parameters['walker_steps']  # Number of steps before walker termination
    flips = parameters['hamming_distance']  # Hamming distance traveled between points

    seed = 42  # Seed for random number generator
    np.random.seed(seed)

    b = random_array(visible_layer_size)  # Visible layer bias
    c = random_array(hidden_layer_size)  # Hidden layer bias
    W = random_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights
    H = generate_positive_ground_state_hamiltonian(visible_layer_size)  # Hamiltonian

    walker = Walker()
    rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters

    walker.random_walk(rbm.probability, flips)

    # Printing results
    print(f"Accept rate: {walker.average_acceptance()}")
    # print(f"Data: {walker.get_walk_results()}" )
    # Plotting histogram with results

    result_list = []

    #rbm = RBM(start_state, visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters

    #for i in range(2 ** bitstring_length):
    #    result_list.append(rbm.probability(int_to_binary_array(i, bitstring_length)))

    # Plotting histogram with results
    norm = sum(result_list)

    # print(100 * (result_list / norm))

    # for i in range(2 ** bitstring_length):
    #    print(rbm.local_energy(H, walker, i))

    history = [state.get_value() for state in walker.get_walk_results()]

    estimate_1 = rbm.get_rbm_energy(walker, H)
    print(f"Estimated ground state 1: {estimate_1}")

    walker = Walker()
    walker.random_walk(rbm.probability, flips)
    estimate_2 = rbm.get_rbm_energy(walker, H)
    print(f"Estimated ground state 2: {estimate_2}")
    print(f"Estimator difference: {np.abs((estimate_1 - estimate_2)/ ((estimate_1+estimate_2)/2)):.2%}")
    print(f"Ground state: {min(np.linalg.eigvals(H))}")

    #print("Optimizing...")
    #res = optimize.minimize(rbm.minimize_energy, rbm.get_variable_array(), (walker, H), options={'disp': True})

    #rbm.set_variables_from_array(res.x)
    #print(f"New estimated ground state: {rbm.get_rbm_energy(walker, H)}")

    # plt.figure(0)
    # plt.hist(history, density=True, bins=2**bitstring_length, edgecolor="black", align='mid')
    # plt.scatter([x for x in range(2**bitstring_length)], (result_list / norm), color='red')
    # plt.show()







