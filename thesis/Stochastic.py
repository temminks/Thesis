import numpy as np


class Stochastic:
    @staticmethod
    def uniform_1(data):
        """Uniform distribution with bounds [d - sqrt(d), d + sqrt(d)].

        d is a a deterministic duration d, i.e. uniformly adding or subtracting
        sqrt(d) from d.

        :param data:
        :return:
        """
        d = np.array(data)
        return np.maximum((d > 0), data + np.random.uniform(-1, 1, size=d.shape) * np.sqrt(data))

    @staticmethod
    def uniform_2(data):
        """Uniform distribution with bounds [0, 2d] for a deterministic duration d.

        :param data:
        :return:
        """
        d = np.array(data)
        return np.maximum((d > 0), np.random.uniform(0, 2, size=d.shape) * data)

    @staticmethod
    def exponential(data):
        """Exponential distribution with expectation d for a deterministic duration d.

        Note from the numpy doc: The scale parameter, β=1/λ

        :param data:
        :return:
        """
        d = np.array(data)
        return np.maximum((d > 0), np.random.exponential(d))

    @staticmethod
    def beta_1(data):
        """Beta distribution on [d/2, 2d] with variance d/3 for a deterministic duration d.

        :param data:
        :return:
        """
        pass

    @staticmethod
    def beta_2(data):
        """Beta distribution on [d/2, 2d] with variance d²/3 for a deterministic duration d.

        :param data:
        :return:
        """
        pass
