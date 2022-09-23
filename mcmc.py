import copy
import random
import numpy as np
from state import State


class Metropolis(object):

    def __init__(self, walker_steps, x: State, distribution):
        self.walker_steps = walker_steps
        self._state_length = x.get_length()
        self.x_new = copy.deepcopy(x)
        self.x_old = State(x.get_length())
        self.distribution = distribution

    def metropolis(self):
        # TODO sjekke forskjell mellom reset av bit_string og ikke, mellom hver kjøring
        accepted = 0
        for i in range(self.walker_steps):
            self.x_new.flip()
            if self.acceptance_criterion(self.distribution):
                self.x_old.set_value(self.x_new.get_value())
                accepted += 1
            else:
                self.x_new.set_value(self.x_old.get_value())
        accept_rate = accepted/self.walker_steps
        return self.x_new, accept_rate

    def acceptance_criterion(self, function, sigma=1.5) -> bool:
        u = random.uniform(0, sigma)

        new_score = self.runOp(function, self.x_new.get_bit_array())
        old_score = self.runOp(function, self.x_old.get_bit_array())

        score = new_score / old_score > u

        return score

    def runOp(self, op, val):
        return op(val)
