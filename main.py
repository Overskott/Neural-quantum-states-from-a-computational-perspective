import matplotlib.pyplot as plt
from mcmc import *
from rbm import RBM
from utils import *
from scipy import optimize


if __name__ == '__main__':

    burn_in_steps = 1000  # Number of steps before collecting points
    walker_steps = 10000  # Number of steps before walker termination
    seed = 42  # Seed for random number generator
    np.random.seed(seed)

    visible = 3  # Number of qubits
    hidden = 6  # Number of hidden nodes

    flips = 1  # Hamming distance traveled between points
    start_state = State(visible)

    low = -1

    b = random_array(visible, low=low)
    c = random_array(hidden, low=low)
    W = random_matrix(visible, hidden, low=low)
    H = generate_positive_ground_state_hamiltonian(visible)

    #b = np.array([-0.78147528, -0.76629846, 0.60323094])
    #c = np.array([0.10772212, -0.09495096, 0.96237605])
    #W = np.array([[-0.99002308, -0.98484, -0.99256982],
    #              [-0.68841895, -0.53552465, -0.64506059],
    #              [-0.26150969, 0.03064657, -0.26203074]])
    #H = np.array([[0.60458211, 0.31568608, 0.12596736],
    #                [0.31568608, 0.19786036, 0.34229307],
    #                [0.12596736, 0.34229307, 0.48115528]])

    walker = Walker(start_state, burn_in_steps, walker_steps)
    rbm = RBM(start_state, visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters

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

    print(f"Estimated ground state: {rbm.get_rbm_energy(walker, H)}")
    walker = Walker(start_state, burn_in_steps, walker_steps)
    walker.random_walk(rbm.probability, flips)
    print(f"Estimated ground state: {rbm.get_rbm_energy(walker, H)}")
    print(f"Ground state: {min(np.linalg.eigvals(H))}")
    #print("Optimizing...")
    #res = optimize.minimize(rbm.minimize_energy, rbm.get_variable_array(), (walker, H), options={'disp': True})

    #rbm.set_variables_from_array(res.x)
    #print(f"New estimated ground state: {rbm.get_rbm_energy(walker, H)}")

    # plt.figure(0)
    # plt.hist(history, density=True, bins=2**bitstring_length, edgecolor="black", align='mid')
    # plt.scatter([x for x in range(2**bitstring_length)], (result_list / norm), color='red')
    # plt.show()







