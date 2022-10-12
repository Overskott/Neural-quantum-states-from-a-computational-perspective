import random

import numpy as np
import matplotlib.pyplot as plt
from rbm import RBM
from state import State
import hamiltonian
from utils import *

low = -1
high = 1
size = 2

H = hamiltonian.random_hamiltonian(size, low, high)
print(H)

print(np.shape(H))