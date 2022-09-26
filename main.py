import matplotlib.pyplot as plt
from mcmc import *
from rbm import RBM


if __name__ == '__main__':

    burn_in_steps = 0
    walker_steps = 1000  # Number of steps before walker termination
    bitstring_length = 8  # Number of qubits
    start_state = State(bitstring_length)
    walker = Walker(start_state, burn_in_steps, walker_steps)
    rbm = RBM(start_state)  # Initializing RBM currently with random configuration and parameters

    walker.random_walk(rbm.probability)

    # Printing results
    print(f"Accept rate: {walker.average_acceptance()}")
    print(f"Data: {walker.get_walk_results()}" )
    # Plotting histogram with results
    plt.hist(walker.get_walk_results())
    plt.show()

    # walkers = 100  # Number of random walkers in total
    # walker_steps = 100  # Number of steps before walker termination
    # bitstring_length = 10  # Number of qubits
    #
    # rbm = RBM(State(bitstring_length))  # Initializing RBM currently with random configuration and parameters
    #
    #
    # walker_list = []  # list to store result of each walker
    # accept_total = 0  # Tracking the acceptance rate for each walker
    #
    # for i in range(walkers):
    #     state = State(bitstring_length)  # Random starting state
    #     met = Metropolis(walker_steps, state, rbm.probability)
    #
    #     run, accept_rate = met.metropolis()
    #     # x_hat = x_hat + run.get_value()
    #     accept_total += accept_rate
    #     walker_list.append(run.get_value())
    #
    # # Printing results
    # print(f"Accept rate: {accept_total / walkers}")
    # # print(f"Probability: {x_hat / walkers}")
    #
    # # Plotting histogram with results
    # plt.hist(walker_list)
    # plt.show()






