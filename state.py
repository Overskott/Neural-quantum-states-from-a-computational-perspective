import random
import numpy as np


class State(object):

    def __init__(self, length: int, value=None):
        self._length = length

        if value is None:
            self._value = self.generate_perm()
        else:
            self._value = value

        self._bit_array = self.value_to_bit_array()

    def __len__(self):
        return self._length

    def __str__(self) -> str:
        return str(self.get_bit_array())

    def get_length(self):
        return self._length

    def get_bit_array(self):
        return self._bit_array

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = int(value)
        self._bit_array = self.value_to_bit_array()

    def value_to_bit_array(self):
        bit_array = np.zeros(self._length)
        binary = format(self._value, 'b')
        index = 1

        for c in binary:
            bit_array[len(binary) - index] = int(c)
            index += 1
            np.flip(bit_array)

        return np.flip(bit_array)  # Flipping to return in 'least significant bit' format

    def bit_array_to_value(self):
        value = 0
        index = self._length - 1
        for bit in self._bit_array:
            value += bit * 2 ** index
            index -= 1

        return value

    def generate_perm(self) -> int:
        """Takes a _length as input, and returns a randomly generated _number with binary _number _length = _length"""
        permutation = random.randint(0, (2 ** self._length)-1)
        return permutation

    def flip(self, flips: int = 1) -> None:
        # TODO fix this for np array
        for i in range(flips):
            flip_index = random.randint(0, self._length - 1)
            self.flip_bit(flip_index)
            self._value = self.bit_array_to_value()

    def flip_bit(self, index):
        bit = self._bit_array[index]
        if bit == 1:
            self._bit_array[index] = 0
        else:
            self._bit_array[index] = 1
