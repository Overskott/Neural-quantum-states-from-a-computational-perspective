import matplotlib.pyplot as plt
import numpy as np

from mcmc import *

if __name__ == '__main__':

    walkers = 100
    walker_steps = 10
    bitstring_length = 5
    sigma = 1.7**bitstring_length
    mu = 2**bitstring_length/2
    np.linspace(-(2**bitstring_length/2), 2**bitstring_length/2)

    normal_dist = lambda x: normal_distribution(x, sigma, mu)


    x_hat = 0
    x_list = []
    y_list = []

    for i in range(walkers):
        state = State(bitstring_length)
        met = Metropolis(walker_steps, state, normal_dist)

        run = met.metropolis()
        x_hat = x_hat + run.get_value()
        plt.plot(run.get_value(), normal_dist(run.get_value()), 'ro')
        plt.plot(run.get_value(), 0, 'b.')
        np.trapz()
    gaus_list = []

    for i in range(2**bitstring_length):
        gaus_list.append(normal_dist(i))



    print('Average: ' + str(x_hat / walkers))
    plt.plot(gaus_list)
    plt.show()