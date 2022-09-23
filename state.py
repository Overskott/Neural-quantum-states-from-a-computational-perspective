import random
from bitstring import Bits, BitArray


class State(object):

    def __init__(self, length: int, value=None):
        self._length = length

        if value is None:
            self._value = self.generate_perm()
        else:
            self._value = value

        self._bitstring = BitArray(Bits(uint=self._value, length=self._length))

    def __len__(self):
        return self._length

    def __str__(self):
        return self.get_bitstring()

    def get_length(self):
        return self._length

    def get_bitstring(self):
        return self._bitstring.bin

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value
        self._bitstring = BitArray(Bits(uint=self._value, length=self._length))

    def generate_perm(self) -> int:
        """Takes a _length as input, and returns a randomly generated _number with binary _number _length = _length"""
        permutation = random.randint(0, (2 ** self._length)-1)
        return permutation

    def flip(self, flips: int = 1) -> None:

        for i in range(flips):
            flip_index = random.randint(0, self._length - 1)
            self._bitstring.invert(flip_index)
            self._value = self._bitstring.uint