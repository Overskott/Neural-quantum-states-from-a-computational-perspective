
from mcmc import *

if __name__ == '__main__':

    def runOp(op, val):
        return op(val)

    # declare full function
    def add(x, y):
        return x+y


    def normal_distribution(x: int, sigma: float, mu: float) -> float:
        """

        :param x:
        :param sigma:
        :param mu:
        """
        _1 = 1 / (sigma * np.sqrt(2 * np.pi))
        _2 = -(1 / 2) * ((x - mu) / sigma) ** 2

        return _1 * np.exp(_2)


    # run example
    def main():
        f = lambda y: add(3, y)
        result = runOp(f, 1) # is 4

    f = lambda y: add(3, y)
    result = runOp(f, 1)  # is 4
    print(result)