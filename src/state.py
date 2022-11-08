import random
import src.utils as utils
from config_parser import get_config_file


class State(object):

    def __init__(self, length: int, value: int = None):
        """

        :param length: State binary array length i.e. number of qubits in the system
        :param value: State value ranging form 0, 2^length - 1
        """
        data = get_config_file()['parameters']  # Load the config file

        self._length = data['visible_size']  # Get number of visible nodes from the config file

        if value is None:
            self._value = int(random.randint(0, (2 ** self._length)-1))
        else:
            self._value = value

        self._bit_array = utils.int_to_binary_array(self._value, self._length)

    def __len__(self):
        return self._length

    def __str__(self) -> str:
        return str(self.get_bit_array())

    def __int__(self) -> int:
        return self._value

    def __lt__(self, other):
        return self._value < other.get_value()

    def __le__(self, other):
        return self._value <= other.get_value()

    def get_length(self):
        return self._length

    def get_bit_array(self):
        return self._bit_array

    def get_value(self) -> int:
        return self._value

    def set_value(self, value):
        self._value = int(value)
        self._bit_array = utils.int_to_binary_array(self._value, self._length)

    def set_bit_array(self, bit_array):
        self._bit_array = bit_array
        self._value = utils.binary_array_to_int(bit_array)

    def flip(self, flips: int = 1) -> None:
        """"""
        for i in range(flips):
            flip_index = random.randint(0, self._length - 1)
            self.flip_bit(flip_index)
            self._value = utils.binary_array_to_int(self._bit_array)

    def flip_bit(self, index):
        """Flips (0->1 or 1->0) the bit on given index of the state"""
        bit = self._bit_array[index]
        if bit == 1:
            self._bit_array[index] = 0
        else:
            self._bit_array[index] = 1
