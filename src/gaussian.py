import matplotlib.pyplot as plt
from mcmc import *
import utils



def normal_distribution(x) -> float:
        """

        :param x:
        :param sigma:
        :param mu:
        """
        sigma = bitstring_length**1
        mu = 16

        if type(x) is not int:
            x = utils.binary_array_to_int(x)

        _1 = 1 / (sigma * np.sqrt(2 * np.pi))
        _2 = -(1 / 2) * ((x - mu) / sigma) ** 2

        return _1 * np.exp(_2)


def double_normal_distribution(x: int, distance: int, sigma_1: float, mu_1: float, sigma_2: float, mu_2: float):

    return (normal_distribution(x, sigma_1, mu_1) + normal_distribution(x+distance, sigma_2, mu_2))/2


if __name__ == '__main__':
    burn_in_steps = 200  # Number of steps before collecting points
    walker_steps = 5000  # Number of steps before walker termination
    bitstring_length = 5  # Number of qubits
    flips = 1  # Hamming distance traveled between points
    start_state = State(bitstring_length)

    sigma = 1
    mu = 2

    normal_dist = lambda x: normal_distribution(x)
    #double_normal_dist = lambda x: double_normal_distribution(x, 400, sigma, mu/2, sigma*2, mu*2)

    walker = Walker(start_state, burn_in_steps, walker_steps)

    walker.random_walk(normal_dist)
    history = [state.get_value() for state in walker.get_walk_results()]

    plt.hist(history, bins=2**bitstring_length, density=True)
    plt.plot([normal_dist(i) for i in range(2**bitstring_length)])

    plt.show()


