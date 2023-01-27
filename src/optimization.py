from config_parser import get_config_file
from src.model import Adam


class GradientDescent(object):

    def __init__(self, parameters, loss_function, learning_rate=None, gradient_steps=None, optimizer=None):
        self.parameters = parameters
        self.loss_function = loss_function
        self.learning_rate = learning_rate

        self.data = get_config_file()['parameters']  # Load the config file

        if learning_rate is None:
            self.learning_rate = self.data['learning_rate']
        else:
            self.learning_rate = learning_rate

        if gradient_steps is None:
            self.gradient_steps = self.data['gradient_descent_steps']
        else:
            self.gradient_steps = gradient_steps

        if optimizer is None:
            self.optimizer = Adam()
        else:
            self.optimizer = optimizer


    def finite_difference(self, index, h=1e-5):

        self.parameters[index] += h
        self.rbm.set_parameters_from_array(params)
        re_plus = self.exact_energy()
        # re_plus = self.estimate_energy()

        params[index] -= 2 * h
        self.rbm.set_parameters_from_array(params)
        re_minus = self.exact_energy()
        # re_minus = self.estimate_energy()
        params[index] += h

        params[index] += h * 1j
        self.rbm.set_parameters_from_array(params)
        im_plus = self.exact_energy()
        # im_plus = self.estimate_energy()

        params[index] -= 2 * h * 1j
        self.rbm.set_parameters_from_array(params)
        im_minus = self.exact_energy()
        # im_minus = self.estimate_energy()

        params[index] += h * 1j

        return (re_plus - re_minus) / (2 * h) + (im_plus - im_minus) / (2 * h * 1j)


    def get_parameter_derivative(self):
        params = self.rbm.get_parameters_as_array()
        params_deriv = []

        for i in range(len(params)):
            params_deriv.append(self.finite_difference(i))

        return params_deriv

    def gradient_descent_step(self):

        try:
            params = params - self.learning_rate * np.array(self.get_parameter_derivative())
            print(f"Params 1: {params}")

           self.optimizer(self.gradient_steps)

            # print(f"Adam optimized grads: {adam_params}")

            self.rbm.set_parameters_from_array(adam_params)

        except KeyboardInterrupt:
            print("Gradient descent interrupted")
            break