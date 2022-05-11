import random


class BitString(object):

    def __init__(self, length: int):
        self.length = length
        self.number = self.generate_perm()

    def __len__(self):
        return self.length

    def __str__(self):
        return self.get_bit_string()

    def get_length(self):
        return self.length

    def get_bit_string(self):
        return bin(self.number)[2:].zfill(self.length)

    def get_number(self):
        return self.number

    def generate_perm(self) -> int:
        """Takes a length as input, and returns a randomly generated number with binary number length = length"""
        return random.randint(0, 2 ** self.length)

    def flip(self):
        binary_string = self.get_bit_string()  # format(self.number, 'b')

        print('Number: ' + str(self.number))
        print('binary string: ' + binary_string)
        print('Len of bs:' + str(len(binary_string)))

        flip_index = random.randint(0, self.length-1)

        print('index: ' + str(flip_index))
        print('index value: ' + binary_string[flip_index])

        if binary_string[flip_index] == '1':
            flipped = binary_string[:flip_index] + '0' + binary_string[flip_index + 1:]
        else:
            flipped = binary_string[:flip_index] + '1' + binary_string[flip_index + 1:]

        print('Flipped: ' + flipped)

        self.number = int(flipped, base=2)
