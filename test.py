from src import utils
from src.rbm import RBM
from src.utils import *

low = -1
high = 1
size = 3

b = np.array([-0.78147528, -0.76629846, 0.60323094])
c = np.array([0.10772212, -0.09495096, 0.96237605])
W = np.array([[-0.99002308, -0.98484, -0.99256982],
             [-0.68841895, -0.53552465, -0.64506059],
             [-0.26150969, 0.03064657, -0.26203074]])


rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)

test_encoding = rbm.get_variable_array()

# print(test_encoding)
test_encoding = test_encoding * 10
rbm.set_variables_from_array(test_encoding)

# print(rbm.b)
# print(rbm.c)
# print(rbm.W)

test_h = utils.generate_positive_ground_state_hamiltonian(size)
print(test_h)
eig, eigvec = np.linalg.eig(test_h)
gs_index = np.argmin(eig)
gs = eigvec[:, gs_index]
gs = gs / (np.sum(gs ** 2))
print(gs)