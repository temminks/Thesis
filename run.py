import json

from ForwardSarsaLambda import ForwardSarsaLambda
from PPO import PPO
from models import models
from project import Project
from stochastic import Stochastic


class J30Runner:
    """Loads the J30 projects."""

    def __init__(self, train=True, stochastic: str = 'uniform_2'):
        self.path = './data/J30/j30'
        self.distributions = [func for func in dir(Stochastic) if
                              callable(getattr(Stochastic, func)) and not func.startswith("__")]
        self.stochastic = stochastic

        with open('./data/J30/train_test_split.txt') as f:
            train_test_split = json.load(f)
            self.train = [self.path + s for s in train_test_split['train']]
            self.test = [self.path + s for s in train_test_split['test']]
            self.projects = self.load_projects(train, stochastic)

    def load_projects(self, train: bool, stochastic: str):
        train_test = self.train if train else self.test
        return [Project(p, stochastic=stochastic) for p in train_test]

    @property
    def stochastic(self):
        return self.__stochastic

    @stochastic.setter
    def stochastic(self, stochastic):
        if stochastic not in self.distributions:
            raise ValueError('{} distribution is not implemented.'.format(stochastic))
        self.__stochastic = stochastic


class ForwardSarsaLambdaRunner(J30Runner):
    def __init__(self, model, stochastic: str = 'uniform_2', episodes=50, gamma=0.95, train=True):
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
        super().__init__(train, stochastic)

        self.episodes = episodes
        self.gamma = gamma
        self.model = model

    def __len__(self):
        return len(self.projects)

    @property
    def gamma(self):
        return self.__gamma

    @property
    def episodes(self):
        return self.__episodes

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


class ForwardSarsaLambdaJ30Runner(ForwardSarsaLambdaRunner):
    """Runner class for the Forward Sarsa(λ) algorithm."""

    def __init__(self, model, train=True, stochastic='uniform_2', episodes=50,
                 gamma=0.95, lam=0.8, epsilon=1.0, eta=0.01):
        super().__init__(model, train, stochastic, episodes, gamma)

        self.lam = lam
        self.epsilon = epsilon
        self.eta = eta

        self.fsl = ForwardSarsaLambda(self.episodes,
                                      self.projects,
                                      self.model,
                                      epsilon=self.epsilon)

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


class PPORunner(J30Runner):
    def __init__(self, episodes, model: models.Model, action_dim, clipping_rate=0.2):
        super().__init__(train=False)

        self.clipping_rate = clipping_rate
        self.PPO = PPO(episodes, self.projects, model, action_dim, clipping_rate)

    @property
    def clipping_rate(self):
        return self.__clipping_rate

    @clipping_rate.setter
    def clipping_rate(self, clipping_rate):
        if not 0 <= clipping_rate <= 1:
            raise ValueError("clipping rate is {} but should be between 0 and 1".
                             format(clipping_rate))
        self.__clipping_rate = clipping_rate
