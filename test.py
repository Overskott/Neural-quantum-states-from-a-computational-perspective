from config_parser import get_config_file
from src import utils
from src.ansatz import RBM
from src.mcmc import Walker
from src.model import Model
from src.utils import *


seed = 44  # Seed for random number generator
np.random.seed(seed)

parameters = get_config_file()['parameters']

visible_layer_size = parameters['visible_size']  # Number of qubits
hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes

b = random_complex_array(visible_layer_size)  # Visible layer bias
c = random_complex_array(hidden_layer_size)  # Hidden layer bias
W = random_complex_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights
#H = random_hamiltonian(2**visible_layer_size)  # Hamiltonian

H =np.array([[-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2]])

walker = Walker()
rbm = RBM(visible_bias=b, hidden_bias=c, weights=W)  # Initializing RBM currently with random configuration and parameters
model = Model(rbm, walker, H)  # Initializing model with RBM and Hamiltonian


print(model.exact_grads(model.rbm.get_parameters_as_array()))
