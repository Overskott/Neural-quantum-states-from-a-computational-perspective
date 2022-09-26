import matplotlib.pyplot as plt
import numpy as np
from mcmc import *



def normal_distribution(x: int, sigma: float, mu: float) -> float:
        """

        :param x:
        :param sigma:
        :param mu:
        """
        _1 = 1 / (sigma * np.sqrt(2 * np.pi))
        _2 = -(1 / 2) * ((x - mu) / sigma) ** 2

        return _1 * np.exp(_2)


def double_normal_distribution(x: int, distance: int, sigma_1: float, mu_1: float, sigma_2: float, mu_2: float):

    return (normal_distribution(x, sigma_1, mu_1) + normal_distribution(x+distance, sigma_2, mu_2))/2


if __name__ == '__main__':

    walkers = 1000
    walker_steps = 100
    bitstring_length = 12
    sigma = 1.4**bitstring_length
    mu = 2**bitstring_length/2
    np.linspace(-(2**bitstring_length/2), 2**bitstring_length/2)

    normal_dist = lambda x: normal_distribution(x, sigma, mu)
    double_normal_dist = lambda x: double_normal_distribution(x, 400, sigma, mu/2, sigma*2, mu*2)

    x_hat = 0
    walker_list = []
    accept_average = 0

    for i in range(walkers):
        state = State(bitstring_length)
        met = Metropolis(walker_steps, state, double_normal_dist)

        run, accept_rate = met.metropolis()
        x_hat = x_hat + run.get_value()
        accept_average += accept_rate
        walker_list.append(run.get_value())

    plt.hist(walker_list, bins=[i for i in range(0, 2 ** bitstring_length, 32)], density=True)
    #plt.plot([normal_dist(i) for i in range(2**bitstring_length)])
    plt.plot([double_normal_dist(i) for i in range(2**bitstring_length)])
    print('Accept rate: ' + str(accept_average / walkers))
    print('E: ' + str(x_hat / walkers))

    plt.legend(["Target", "MCMC"], loc="upper right")
    plt.show()


