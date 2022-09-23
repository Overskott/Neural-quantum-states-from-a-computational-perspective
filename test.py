import numpy as np
import random


sigma = np.asarray([random.randint(0, 1) for _ in range(10)]).reshape(-1, 1)

print(sigma)
