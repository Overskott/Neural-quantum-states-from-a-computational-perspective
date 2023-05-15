import copy
import random
import numpy as np
from config_parser import get_config_file
from src import utils


class Walker(object):

    def __init__(self,
                 visible_size: int,
                 steps: int,
                 burn_in: int,
                 hamming_distance: int = 1,
                 ):

        self.steps = steps
        self.burn_in = burn_in
        self.hamming_distance = hamming_distance
        self.current_state = np.random.randint(0, 2, visible_size)
        self.next_state = copy.deepcopy(self.current_state)
        self.walk_results = []
        self.acceptance_rate = 0

    def __call__(self, function, num_steps):

        self.estimate_distribution(function)
        return np.asarray(self.get_history())

    def get_steps(self):
        return self.steps

    def get_history(self):
        return self.walk_results

    def clear_history(self):
        self.walk_results = []

    def estimate_distribution(self, function, burn_in=True) -> None:
        self.clear_history()

        if burn_in:
            self.burn_in_walk(function)

        self.random_walk(function)

    #@profile
    def random_walk(self, function):

        for i in range(self.steps):
            self.next_state = utils.hamming_step(self.current_state)
            self.walk_results.append(self.current_state)

            if self.acceptance_criterion(function):
                self.current_state = copy.deepcopy(self.next_state)
                self.acceptance_rate += 1

            else:
                self.next_state = copy.deepcopy(self.current_state)

    def burn_in_walk(self, function):
        for i in range(self.burn_in):
            self.next_state = utils.hamming_step(self.current_state)

            if self.acceptance_criterion(function):
                self.current_state = copy.deepcopy(self.next_state)
            else:
                self.next_state = copy.deepcopy(self.current_state)

    def average_acceptance(self):
        return self.acceptance_rate / self.steps

    #@profile
    def acceptance_criterion(self, function) -> bool:
        u = random.uniform(0, 1)
        new_score = function(self.next_state)
        old_score = function(self.current_state)

        score = new_score / old_score > u

        return score

    # def hamming_step(self, flips: int = 1) -> None:
    #
    #     used_indexes = []
    #     for i in range(flips):
    #         flip_index = random.randint(0, self.current_state.n-1) # minus 1?
    #
    #         while flip_index in used_indexes:
    #             flip_index = random.randint(0, self.current_state.n-1)
    #
    #         used_indexes.append(flip_index)
    #         self.next_state[flip_index] = 1 - self.next_state[flip_index]
    #         #self.flip_bit(flip_index)


