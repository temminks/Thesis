import json

from thesis.ForwardSarsaLambda import ForwardSarsaLambda
from thesis.Project import Project
from thesis.Stochastic import Stochastic


class Runner:
    def __init__(self, model, stochastic: str = 'uniform_2', episodes=50, gamma=0.95):
        """Initialise a new wrapper to run algorithms.

        :param model:
        :param stochastic: the kind of stochastic distribution to use for the
        duration of the tasks
        :param episodes: number of episodes, one episode is a run on all training
        or test projects
        :param gamma: discount factor γ ϵ [0, 1] that controls how future
        rewards are weighted with respect to the immediate reward. Usually a
        future reward is perceived as less valuable compared to an immediate
        reward of the same value, i.e. a rewards of 1 now is more valuable than
        a rewards of 1 in the future.
        """
        self.episodes = episodes
        self.gamma = gamma
        self.model = model

        self.distributions = [func for func in dir(Stochastic) if
                              callable(getattr(Stochastic, func)) and not func.startswith("__")]

        self.stochastic = stochastic
        self.train = self.test = self.projects = None

    def __len__(self):
        return len(self.projects)

    @property
    def gamma(self):
        return self.__gamma

    @property
    def episodes(self):
        return self.__episodes

    @property
    def stochastic(self):
        return self.__stochastic

    @stochastic.setter
    def stochastic(self, stochastic):
        if stochastic not in self.distributions:
            raise ValueError('{} distribution is not implemented.'.format(stochastic))
        self.__stochastic = stochastic

    @episodes.setter
    def episodes(self, episodes):
        if not 0 <= episodes:
            raise ValueError("the number of episodes should be larger than 0")
        self.__episodes = episodes

    @gamma.setter
    def gamma(self, gamma):
        if not 0 <= gamma <= 1:
            raise ValueError("discount factor γ is {} but should be between 0 and 1".
                             format(gamma))
        self.__gamma = gamma


class J30Runner(Runner):
    """Loads the J30 projects."""

    def __init__(self, model, train=True, stochastic='uniform_2', episodes=50, gamma=0.95):
        super().__init__(model, stochastic, episodes, gamma)

        self.path = './data/J30/j30'

        with open('./data/J30/train_test_split.txt') as f:
            train_test_split = json.load(f)
            self.train = [self.path + s for s in train_test_split['train']]
            self.test = [self.path + s for s in train_test_split['test']]
            self.load_projects(train, stochastic)

    def load_projects(self, train: bool, stochastic: str):
        train_test = self.train if train else self.test
        self.projects = [Project(p, stochastic=stochastic) for p in train_test]


class ForwardSarsaLambdaRunner(J30Runner):
    """Runner class for the Forward Sarsa(λ) algorithm."""

    def __init__(self, model, train=True, stochastic='uniform_2', episodes=50, gamma=0.95,
                 lam=0.8, epsilon=1.0, eta=0.01, model_name=None):
        super().__init__(model, train, stochastic, episodes, gamma)

        self.lam = lam
        self.epsilon = epsilon
        self.eta = eta
        self.model_name = model_name

        self.fsl = ForwardSarsaLambda(self.episodes,
                                      self.projects,
                                      self.model,
                                      epsilon=self.epsilon,
                                      model_name=model_name)

    @property
    def lam(self):
        return self.__lam

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if not 0 <= epsilon <= 1:
            raise ValueError("greedy parameter ϵ is {} but should be between 0 and 1".
                             format(epsilon))
        self.__epsilon = epsilon

    @lam.setter
    def lam(self, lam):
        if not 0 <= lam <= 1:
            raise ValueError("trace-decay parameter λ is {} but should be between 0 and 1".
                             format(lam))
        self.__lam = lam
