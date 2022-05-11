import random


class BitString(int):

    def __init__(self, length: int):
        self.length = length
        self.number = self.generate_perm()

    def get_length(self):
        return self.length

    def generate_perm(self) -> int:
        """Takes a length as input, and returns a randomly generated number with binary number length = length"""

        return random.randint(0, 2 ** self.length)

    def flip(self, x_old: int):

        binary_string = format(x_old, 'b')
        flip_index = random.randint(0, self.length) - 1

        print('Number: ' + str(x_old))
        print('binary string: ' + binary_string)
        print('Len of bs:' + str(len(binary_string)))
        print('index: ' + str(flip_index))
        print('index value: ' + binary_string[flip_index])

        if binary_string[flip_index] == '1':
            flipped = binary_string[:flip_index] + '0' + binary_string[flip_index + 1:]
        else:
            flipped = binary_string[:flip_index] + '1' + binary_string[flip_index + 1:]

        print('Flipped: ' + flipped)

        return int(flipped, base=2)

    def get_bit_string(self):
        return format(self.number, 'b')
