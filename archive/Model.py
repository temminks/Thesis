from archive.Activation import *


class Model:
    """This is the model or architecture of the neural network. This class
    uses the Adam optimiser as an improves method of stochastic gradient
    descent.

    Adam in numpy
    https://github.com/jrios6/Adam-vs-SGD-Numpy/blob/master/Adam%20vs%20SGD%20-%20On%20Kaggle's%20Titanic%20Dataset.ipynb
    """

    def __init__(self, layer_dims=None, activation='relu', learning_rate=0.001, parameters=None):
        """Description

        Parameters
        ----------
        layer_dims : list
            A list that contains the structure, i.e. the number of nodes per layer,
            of the neural network. Each layer is a fully connected layer followed
            by dropout layer.
        activation : str
        learning_rate : float
        parameters : dictionary
        Example
        -------
        Model(layers=[3, 4, 1], learning_rate=0.001)
        results in a two layer neural network with three input nodes, four nodes in
        the hidden layer (first layer) and one output node (second layer).
        """
        self.L = len(layer_dims)
        try:
            self.activation = getattr(Activation, activation)
        except AttributeError:
            print("Only 'relu', 'leakyRelu', 'sigmoid' and 'none' activation",
                  "functions are implemented. \nYou can implement '{}' in the",
                  "'Activation' class.".format(activation))
            raise AttributeError

        self.learning_rate = learning_rate
        self.first_moment = [0] * (self.L - 1)  # mean estimate
        self.second_moment = [0] * (self.L - 1) # uncentered variance
        self.first_moment_bias = [0] * (self.L - 1)
        self.second_moment_bias = [0] * (self.L - 1)

        # epsilon used for numerical stability
        self.epsilon = 10e-8

        self.parameters = {}
        if not parameters:
            for l in range(1, self.L):
                self.parameters['W' + str(l)] = self.initialiser(layer_dims[l-1], layer_dims[l])
                self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        else:
            self.parameters = parameters

    @staticmethod
    def initialiser(num_inputs, num_outputs):
        """Weight initialisation according to He et al.

        He et al. propose to use this weight initilisation instead of Xavier
        for ReLU activation functions which yields Var(w_l) = 2 * n_l
        cf. https://arxiv.org/abs/1502.01852, pages 3-4

        :param num_inputs: Number of input nodes
        :param num_outputs: Number of output nodes
        :return: Weights for a single neuron.
        """
        return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / num_inputs),
                                size=(num_outputs, num_inputs))

    def forward_pass(self, inputs, weights, bias):
        """The forward pass in the neural network, i.e. activation(w * X + b)
        :param inputs:
        :param weights:
        :param bias:
        :return:
        """
        z = np.dot(weights, inputs) + bias
        return self.activation(z), [(inputs, weights, bias), z]

    def predict(self, X):
        ''' Predicts the output for the input X. The input X can be interpreted
        as the activation output of the 0th layer.

        Parameters
        ----------
        X: int, list, numpy.ndarray
            input vector

        Returns
        -------
        numpy.ndarray
            The output of the neural network.
        dictionary
            The cache of the parameters X, W, b ("linear cache") as well as the output of the
            linear function z = wx + b
        '''
        cache = {}
        A = X

        for l in range(1, self.L):
            A, cache[l] = self.forward_pass(A, self.parameters["W" + str(l)],
                                            self.parameters["b" + str(l)], self.activation)
        return A, cache

    def train(self, target, AL, cache, beta_1=0.9, beta_2=0.999, t=1):
        """Backpropagation

        :param target:
        :param AL:
        :param cache:
        :param beta_1: Momentum parameter
        :param beta_2: RMSProp parameter
        :param t:
        :return:
        """
        gradients = self.gradients(target, AL, cache)
        for l in range(self.L - 1, 0, -1):
            self.first_moment[l-1] = self.first_moment[l-1] * beta_1 + (1 - beta_1) * \
                                     gradients['dW' + str(l+1)]
            self.first_moment_bias[l-1] = self.first_moment_bias[l-1] * beta_1 + (1 - beta_1) * \
                                          gradients['db' + str(l+1)]
            self.second_moment[l-1] = self.second_moment[l-1] * beta_2 + (1 - beta_2) * \
                                      (gradients['dW' + str(l+1)] ** 2)
            self.second_moment_bias[l-1] = self.second_moment_bias[l-1] * beta_2 + (1 - beta_2) * \
                                           (gradients['db' + str(l+1)] ** 2)

            # bias correction
            first_moment_corrected = self.first_moment[l - 1] / (1 - (beta_1 ** t))
            first_moment_bias_corrected = self.first_moment_bias[l - 1] / (1 - (beta_1 ** t))
            second_moment_corrected = self.second_moment[l - 1] / (1 - (beta_2 ** t))
            second_moment_bias_corrected = self.second_moment_bias[l - 1] / (1 - (beta_2 ** t))

            weight_update = first_moment_corrected / (np.sqrt(second_moment_corrected) + self.epsilon)
            bias_update = first_moment_bias_corrected / (np.sqrt(second_moment_bias_corrected) + self.epsilon)
            self.parameters['W' + str(l)] -= self.learning_rate * weight_update
            self.parameters['b' + str(l)] -= self.learning_rate * bias_update

    def gradients(self, target, AL, cache):
        """Computes the gradients for each layer.

        Parameters
        ----------
        target: int
        AL: int
        cache: dictionary

        Returns
        -------
        """
        grads = {}
        current_cache = cache[self.L - 1]
        dZ = self.derror(target, AL) * self.activation(AL, current_cache[1], derivative=True)
        grads["dA" + str(self.L-1)], grads["dW" + str(self.L)], grads["db" + str(self.L)] = self.linear_backwards(
            dZ, current_cache[0])

        for l in range(self.L - 2, 0, -1):
            current_cache = cache[l]
            grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = self.backwards(
                                                grads["dA" + str(l + 1)], current_cache)
        return grads

    def backwards(self, dA, cache):
        """dZ = dA * g'(Z) for dZ, dA and Z in a layer l with activation function g(·)"""
        dZ = self.activation(dA, cache[1], derivative=True)
        return self.linear_backwards(dZ, cache[0])

    @staticmethod
    def linear_backwards(dZ, linear_cache):
        """Backward propagation part of the linear function.

        :param dZ:
        :param linear_cache:
        :return: dA_prev, dW, db
        """
        A_prev, W, b = linear_cache
        m = len(A_prev)

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    @staticmethod
    def derror(target, AL, etype='MSE'):
        """Calculates the derivative of the error (or cost).

        Parameters
        ----------
        target:
        AL:
        etype: str
            error type, MSE for mean squared error, CE for cross entropy
        """
        if etype == 'MSE':
            return target - AL
        elif etype == 'CE':
            return - (np.divide(target, AL) - np.divide(1 - target, 1 - AL))