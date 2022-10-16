
import random
import numpy as np




class Hamiltonian():

    def __init__(self, size):
        self.size = size
        self.h_max = 2
        self.h = [random.uniform(-1, 1)*self.h_max for _ in range(self.size)]


