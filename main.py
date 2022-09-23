import matplotlib.pyplot as plt
from mcmc import *
from rbm import RBM


if __name__ == '__main__':

    walkers = 1000
    walker_steps = 50
    bitstring_length = 30

    start_state = State(bitstring_length)
    rbm = RBM(start_state)

    x_hat = 0
    walker_list = []
    accept_average = 0

    for i in range(walkers):
        state = State(bitstring_length)
        met = Metropolis(walker_steps, state, rbm.probability)

        run, accept_rate = met.metropolis()
        x_hat = x_hat + run.get_value()
        accept_average += accept_rate
        walker_list.append(run.get_value())

    plt.hist(walker_list)
    print(f"Accept rate: {accept_average / walkers}")
    print('E: ' + str(x_hat / walkers))

    plt.legend(["Target", "MCMC"], loc="upper right")
    plt.show()


