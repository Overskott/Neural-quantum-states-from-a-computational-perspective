# This is a sample Python script.
import random
import numpy as np
import matplotlib.pyplot as plt

n_steps=1

def random_bit_string(length: int)-> int:
    """Takes a length as input, and returns a randomly generated number with binary number length = length"""

    return random.randint(0, 2**length)


def normal_distribution(x: int, sigma: float, mu: float) -> float:
    """

    :param x:
    :param sigma:
    :param mu:
    """
    _1 = 1/(sigma*np.sqrt(2*np.pi))
    _2 = -(1/2)*((x-mu)/sigma)**2
    return _1*np.exp(_2)


def flip(x_old: int):
    print('Number: ' + str(x_old))
    binary_string = format(x_old, 'b')
    print('binary string: ' + binary_string)
    print('Len of bs:' + str(len(binary_string)))
    flip_index = random.randint(0, len(binary_string)-1)
    print('index: ' + str(flip_index))
    print('index value: ' + binary_string[flip_index])


    if binary_string[flip_index] == '1':
        flipped = binary_string[:flip_index] + '0' + binary_string[flip_index + 1:]
    else:
        flipped = binary_string[:flip_index] + '1' + binary_string[flip_index + 1:]
    print('Flipped: ' + flipped)

    return int(flipped, base=2)


def find_x_new(x_old):
    return flip(x_old)

def acceptance_criterion(x_new, x_old, P) -> bool:
    u = random.uniform(0, 1)

    return (P(x_new)/P(x_old)) > u

def metropolis(P, n_steps, length):

    x_old = random_bit_string(length) # TODO sjekke forskjell mellom reset av x_old og ikke, mellom hver kjøring

    for i in range(n_steps):
        x_new = find_x_new(x_old)
        if acceptance_criterion(x_new, x_old, P):
           x_old = x_new
    return x_old

def average(N):
    X_hat = 0
    for i in range(N):
        X_hat = X_hat + metropolis()
    return X_hat/N



if __name__ == '__main__':

    # n = 200
    #
    # dist_list = np.zeros((n, 2))
    #
    # for i in range(n):
    #     point = random_bit_string(5)
    #
    #     dist_list[i, 0] = point
    #     dist_list[i, 1] = normal_distribution(point, 3, 15)
    #
    #     plt.plot(point, normal_distribution(point, 3, 15), 'rx')
    #
    # plt.show()
    # dist_list.sort()
    # plt.plot(dist_list[:, 1], dist_list[:, 0])
    #
    # plt.show()

    n= 16

    print(flip(n))
