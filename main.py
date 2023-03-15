import matplotlib.pyplot as plt



from src.ansatz import RBM
from src.mcmc import *
from src.model import Model
from src.utils import *
from src.hamiltonians import Hamiltonian, IsingHamiltonian, ReducedIsingHamiltonian, DiagonalHamiltonian

if __name__ == '__main__':

    parameters = get_config_file()['parameters']
    visible_layer_size = parameters['visible_size']  # Number of qubits
    hidden_layer_size = parameters['hidden_size']  # Number of hidden nodes

    seed = 44  # Seed for random number generator
    np.random.seed(seed)

    b = random_complex_array(visible_layer_size)  # Visible layer bias
    c = random_complex_array(hidden_layer_size)  # Hidden layer bias
    W = random_complex_matrix(visible_layer_size, hidden_layer_size)  # Visible - hidden weights
    rbm = RBM(visible_bias=b, hidden_bias=c,
              weights=W)  # Initializing RBM currently with random configuration and parameters
    H = Hamiltonian(visible_layer_size)

    walker = Walker()

    model = Model(rbm, walker, H)  # Initializing model with RBM and Hamiltonian
    model_copy = copy.deepcopy(model)

    # Printing results
    print(f"Accept rate: {model.walker.average_acceptance()}")
    #print(f"Estimated energy: {model.estimate_energy()}")
    #print(f"Exact energy: {np.linalg.eigvalsh(H)}")
    #print(f"Ground state: {np.linalg.eig(H)[1].T[0]}")

    fd_plot_list = model.gradient_descent(gradient_method='finite_difference', exact_dist=False)
    #fd_plot_list_mc = model.gradient_descent(gradient_method='finite_difference', exact_dist=False)
    #analytic_plot_list = model.gradient_descent(gradient_method='analytical', exact_dist=True)
    #analytic_plot_list_mc = model_copy.gradient_descent(gradient_method='analytical', exact_dist=False)

    print(f"Optimization time FD: {model.optimizing_time}")
    print(f"Optimization time Analytic: {model_copy.optimizing_time}")
    print(f"Estimated energy FD: {model.estimate_energy()}")
    print(f"Estimated energy Analytic: {model_copy.estimate_energy()}")
    print(f"Exact energies: {np.linalg.eigvalsh(H)}")

    plt.plot(np.real(fd_plot_list), label='Finite Difference')
    #plt.plot(np.real(analytic_plot_list), label='Analytical')
    #plt.plot(np.real(fd_plot_list_mc), label='Finite Difference MC')
    #plt.plot(np.real(analytic_plot_list_mc), label='Analytical MC')

    #plt.axhline(y=min(np.linalg.eigvalsh(H)), color='red', linestyle='--', label='Ground State')
    plt.title('Analytical vs finite difference gradient descent')
    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()






