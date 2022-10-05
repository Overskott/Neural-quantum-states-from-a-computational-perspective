import copy
import random

import numpy as np

from state import State


class Walker(object):

    def __init__(self, state: State, burn_in_steps, steps):
        self.burn_in = burn_in_steps
        self.steps = steps
        self.walk_results = []
        self.current_state = state
        self.next_state = copy.deepcopy(self.current_state)
        self.acceptance_rate = 0

    def get_steps(self):
        return self.steps

    def get_walk_results(self):
        return self.walk_results

    def random_walk(self, function, flips=1):

        for i in range(self.burn_in):
            self.next_state.flip()

            if self.acceptance_criterion(function):
                self.current_state.set_value(self.next_state.get_value())
            else:
                self.next_state.set_value(self.current_state.get_value())

        for i in range(self.steps):
            self.next_state.flip()
            self.walk_results.append(copy.deepcopy(self.current_state))

            if self.acceptance_criterion(function):
                self.current_state.set_value(self.next_state.get_value())
                self.acceptance_rate += 1

            else:
                self.next_state.set_value(self.current_state.get_value())

        return self.next_state

    def average_acceptance(self):
        return self.acceptance_rate / self.steps

    def acceptance_criterion(self, function) -> bool:
        u = random.uniform(0, 1)

        new_score = function(self.next_state.get_bit_array())
        old_score = function(self.current_state.get_bit_array())

        score = new_score / old_score > u

        return score


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

    def acceptance_criterion(self, function, sigma=1) -> bool:
        u = random.uniform(0, sigma)

        new_score = function(self.x_new.get_bit_array())
        old_score = function(self.x_old.get_bit_array())

        score = new_score / old_score > u

        return score
