# This is a sample Python script.
import random

n_steps=1

def random_bit_string(length: int):

def find_x_new(x_old):
    return flip(x_old)

def acceptance_criterion(x_new, x_old, P) -> bool:
    u = random.uniform(0,1)

    return (P(x_new)/P(x_old)) > u

def metropolis(P, n_steps, length):

    x_old = random_bit_string(length) # TODO sjekke forskjell mellom reset av x_old og ikke mellom hver kjøring

    for i in range(n_steps):
        x_new = find_x_new(x_old)
        if acceptance_criterion(x_new, x_old, P):
           x_old = x_new
    return x_old

def average(N):
    X_hat = 0
    for i in range(N):
        X_hat = X_hat + metropolis
    return X_hat/N
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
