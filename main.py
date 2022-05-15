import matplotlib.pyplot as plt
import numpy as np

from mcmc import *

if __name__ == '__main__':

    walkers = 100
    walker_steps = 10
    bitstring_length = 12
    sigma = 1.5**bitstring_length
    mu = 2**bitstring_length/2
    np.linspace(-(2**bitstring_length/2), 2**bitstring_length/2)

    normal_dist = lambda x: normal_distribution(x, sigma, mu)


    x_hat = 0
    x_list = []
    y_list = []
    accept_average = 0

    for i in range(walkers):
        state = State(bitstring_length)
        met = Metropolis(walker_steps, state, normal_dist)

        run, accept_rate = met.metropolis()
        x_hat = x_hat + run.get_value()
        accept_average += accept_rate

        plt.plot(run.get_value(), normal_dist(run.get_value()), 'r.')
        plt.plot(run.get_value(), 0, 'b.')

    gauss_list = []

    for i in range(2**bitstring_length):
        gauss_list.append(normal_dist(i))

    print('Accept rate: ' + str(accept_average/walkers))
    print('E: ' + str(x_hat / walkers))

    plt.plot(gauss_list)

    plt.legend(["MCMC", "spread"], loc="upper right")
    plt.show()