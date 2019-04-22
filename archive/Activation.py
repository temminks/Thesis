import numpy as np


class Activation:
    @staticmethod
    def sigmoid(Z, activation_cache=None, derivative=False):
        """The sigmoid function is an S-shaped function with values between 0 and 1.
        It is often used to predict the probability of an output."""
        if derivative:
            s = 1 / (1 + np.exp(-activation_cache))
            return Z * s * (1 - s)
        else:
            return 1 / (1 + np.exp(-Z))

    @staticmethod
    def relu(Z, activation_cache=None, derivative=False, leakage=0.0):
        """Rectified Linear Unit activation function.

        The activation value of an input Z is the maximum of 0 and Z:
        g(Z)=max(0, Z). The derivative g'(Z) is 1 for Z>0 and otherwise 0.
        Technically, the derivative for g'(0) does not exist, due to the kink
        in g(·) but is usually set to either 0 or 1.

        :param Z:
        :param activation_cache: used to efficiently calculate the derivative
        :param derivative: whether to compute g(·) or the derivative g'(·)
        :param leakage: set leakage to 0.01 for a leakyRelu activation function
        or some other small positive value
        :return:
        """
        if derivative:
            if type(Z) == int:
                return leakage if Z <= 0 else 1
            else:
                dZ = np.array(Z, copy=True)
                assert (dZ.shape == activation_cache.shape)

                dZ[activation_cache <= 0] = leakage
                dZ[activation_cache > 0] = 1
                return dZ
        else:
            return np.maximum(leakage * Z, Z)

    @staticmethod
    def leakyRelu(Z, activation_cache=None, derivative=False):
        """Leaky Rectified Linear Unit with leaky slope 0.01.

        An attempt to prevent zero gradients with ReLU by increasing the range.
        The derivative g'(Z) is 1 for Z>0 and otherwise 0. Technically, the
        derivative for g'(0) does not exist, due to the kink in g(·) but is
        usually set to either 0.01 or 1.

        :param Z:
        :param activation_cache:
        :param derivative:
        :return:
        """
        return Activation.relu(Z, activation_cache, derivative, leakage=0.01)

    @staticmethod
    def none(Z, **kwargs):
        return Z
