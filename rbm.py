import random
import numpy as np

class RBM(object):
n = 10  # Number of nodes in the visible layer
s = np.asarray([random.randint(0, 1) for _ in range(n)])  # Visible layer state
h = np.asarray([random.randint(0, 1) for _ in range(n)])  # Hidden layer state
b = np.random.rand(n)  # Visible layer bias
c = np.random.rand(n)  # Hidden layer bias
W = np.random.rand(n, n)  # s - h weights
z = n  # normalization TODO Ask about this!


def energy():
    '''Calculates the RBM energy'''
    e = 0
    e += np.transpose(h) @ W @ s
    e += np.transpose(c) @ h
    e += np.transpose(b) @ s
    return e


def probability():
    '''Calculates the probability of finding the RBM in state s'''  #TODO is it the probability of findig s?
    product = 1

    for i in range(n):
        scalar = (W[i, :] @ np.transpose(s))
        product *= (1 + np.exp(-scalar - c[i]))

    bias = np.exp(np.transpose(b) @ s)

    return 1/z * product * bias


for _ in range(10):
    s = np.asarray([random.randint(0, 1) for _ in range(n)])  # Visible layer state

    print(f"State: {s}")
    print(f"P = {probability()}")
    print(f"E = {energy()}")