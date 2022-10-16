import random

import numpy as np
import matplotlib.pyplot as plt

import utils
from rbm import RBM
from state import State
import hamiltonian
from utils import *

low = -1
high = 1
size = 3

b = np.array([-0.78147528, -0.76629846, 0.60323094])
c = np.array([0.10772212, -0.09495096, 0.96237605])
W = np.array([[-0.99002308, -0.98484, -0.99256982],
             [-0.68841895, -0.53552465, -0.64506059],
             [-0.26150969, 0.03064657, -0.26203074]])

start_state = State(size)
rbm = RBM(start_state, visible_bias=b, hidden_bias=c, weights=W)
theta_array = utils.create_variable_array(rbm)

print(theta_array)