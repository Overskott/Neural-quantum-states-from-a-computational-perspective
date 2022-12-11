import copy
import random
import numpy as np
from config_parser import get_config_file
import src.utils as utils

from src.state import State


class Walker(object):

    def __init__(self):

        data = get_config_file()['parameters']  # Load the config file

        self.burn_in = data['burn_in_steps']
        self.steps = data['walker_steps']
        self.walk_results = []
        self.current_state = np.random.randint(0, 2, data['visible_size'])
        self.next_state = copy.deepcopy(self.current_state)
        self.acceptance_rate = 0

    def get_steps(self):
        return self.steps

    def get_history(self):
        return self.walk_results

    def clear_history(self):
        self.walk_results = []

    def random_walk(self, function, flips=1):

        for i in range(self.burn_in):
            self.hamming_step(flips)

            if self.acceptance_criterion(function):
                self.current_state = copy.deepcopy(self.next_state)
            else:
                self.next_state = copy.deepcopy(self.current_state)

        for i in range(self.steps):
            self.hamming_step(flips)
            self.walk_results.append(self.current_state)

            if self.acceptance_criterion(function):
                self.current_state = copy.deepcopy(self.next_state)
                self.acceptance_rate += 1

            else:
                self.next_state = copy.deepcopy(self.current_state)

    def average_acceptance(self):
        return self.acceptance_rate / self.steps

    def acceptance_criterion(self, function) -> bool:
        u = random.uniform(0, 1)

        new_score = function(self.next_state)
        old_score = function(self.current_state)

        score = new_score / old_score > u

        return score

    def hamming_step(self, flips: int = 1) -> None:

        used_indexes = []
        for i in range(flips):
            flip_index = random.randint(0, self.current_state.size-1) # minus 1?

            while flip_index in used_indexes:
                flip_index = random.randint(0, self.current_state.size-1)
                print("flip_index", flip_index)

            used_indexes.append(flip_index)
            self.next_state[flip_index] = 1 - self.next_state[flip_index]
            #self.flip_bit(flip_index)

    def flip_bit(self, index):
        """Flips (0->1 or 1->0) the bit on given index of the state"""
        self.next_state[index] = 1 - self.next_state[index]
