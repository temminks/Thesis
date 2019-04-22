import logging
from collections import deque

import numpy as np

from thesis.Project import Project


class ForwardSarsaLambda:
    """Implementation of the Forward Sarsa(λ) algorithm.

    This algorithm is based on the true online Sarsa(λ) algorithm by
    van Seijen, Harm; Sutton, Richard S. but a more efficient implementation
    to use with deep neural networks.

    Some related implementations:
    * Episodic Semi-gradient Sarsa for Estimating q using NN with Keras:
        * https://gist.github.com/FlashTek/0dfddf46c4d50c4e068f1ecbad1d03b5
        * https://chat.stackoverflow.com/rooms/150499/discussion-between-flashtek-and-neil-slater
        * https://stackoverflow.com/questions/45377404/episodic-semi-gradient-sarsa-with-neural-network/45391630#45391630
    * Sarsa Keras
        * https://github.com/keras-rl/keras-rl/blob/master/rl/agents/sarsa.py
    * Python implementations of Sutton, Barto (2018) (incl. TD(λ))
        * https://github.com/ShangtongZhang/reinforcement-learning-an-introduction
        * https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/ed1ddb0c40e37a8846665f26f01e0522b232634f/chapter12/RandomWalk.py
    * Sarsa(λ)
        * https://github.com/codebox/sarsa-lambda/blob/master/strategy.py
    * TD gammon
        * https://github.com/fomorians/td-gammon/blob/master/model.py
    * True Online TD Lambda with Fourier Basis
        * https://github.com/EllaBot/true-online-td-lambda/blob/master/true_online_td_lambda/true_online_td_lambda.py
    """

    def __init__(self, episodes, projects: list[Project], model, lam=0.8,
                 gamma=0.95, eta=0.01, epsilon=1.0, model_name=None):
        """ Initialises the algorithm's parameters.

        :param episodes:
        :param projects:
        :param model: Class with a Keras or TensorFlow model named model
        :param lam: The trace-decay parameter λ ϵ [0, 1] determines the rate
        at which the eligibility trace decays.
                λ = 0 results in a one-step TD return
                λ = 1 results in Monte Carlo behaviour
        :param eta: η will typically be set to 0.01 or larger
        :param epsilon: greediness or exploration parameter ϵ ϵ [0, 1]. ϵ should
        start at a high value close or exactly 1 and decrease.
        :param model_name:
        """

        self.lam = lam
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes * len(projects)

        self.K = int(np.ceil(np.log(eta) / np.log(gamma * lam)))
        self.model = model.model
        self.c_final = (gamma * lam) ** (self.K - 1)
        self.F = deque(maxlen=self.K)
        self.projects = projects
        self.project_id = 0
        self.project = projects[self.project_id]
        self.num_of_tasks = self.project.num_of_tasks
        self.num_of_parameters = 242  # TODO maybe change that into a function
        self.tensorboard = model.tensorboard
        self.model_name = model_name if model_name else "".join(
            np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 7))
        self.topology = self.project.adjacency()

    def __repr__(self):
        # TODO update the representation to fit the final version of the algorithm
        return "TrueOnlineSarsaLambda({}, {})".format(self.lam, self.gamma)

    def train(self):
        """Uses the Forward Sarsa(λ) algorithm to train the model."""
        for episode in range(self.episodes):
            if ((episode + 1) % len(self.projects)) == 0:
                logging.info('episode: {}'.format((episode + 1) / len(self.projects)))
                self.model.save_weights('.\\models\\' + self.model_name + '-' +
                                        str(int(episode / len(self.projects))) + '.h5')
            target_sync = 0
            i = 0
            c = 1
            v_current = 0
            ready = False
            t = 0

            while not self.project.is_finished():
                value, best_action, durations, state = self.act()
                t += self.project.next(best_action, durations)

                if self.project.is_finished():
                    value_next = 0
                    reward = 1 / t
                else:
                    value_next, action_next, durations_next, _ = self.act()
                    reward = 0

                rho = reward + self.gamma * (1 - self.lam) * value_next  # see equation (7)
                self.F.appendleft((state, rho))  # push tuple ‹S, ρ› on F
                delta = reward + self.gamma * value_next - value

                '''"We can avoid these blow-ups in an elegant way, by recomputing the K-bounded
                λ-return from scratch every K time steps.", cf. p. 6'''
                if i == self.K - 1:
                    target = target_sync
                    target_sync = v_current
                    i = 0
                    c = 1
                    ready = True
                else:
                    target_sync += c * delta
                    i += 1
                    c *= (self.gamma * self.lam)

                if ready:
                    target += self.c_final * delta
                    state_pop, rho_pop = self.F.pop()
                    self.model.fit(x=state_pop, y=target, verbose=0)
                    if self.K != 1:
                        target = (target - rho_pop) / (self.gamma * self.lam)

            # results in a 5% epsilon in the last episode
            self.epsilon = self.epsilon * np.exp(-(2.99573227355 / self.episodes))

            if not ready:
                target = target_sync

            while len(self.F) > 0:
                value_pop, rho_pop = self.F.pop()
                callbacks = [self.tensorboard] if len(self.F) == 1 else None
                self.model.fit(x=value_pop, y=target, verbose=0, callbacks=callbacks)
                if self.K != 1:
                    target = (target - rho_pop) / (self.gamma * self.lam)

            logging.debug('epsilon:', int(self.epsilon))

            self.next_project()

    def next_project(self):
        """Reset the current project and switch the project to the next one."""
        self.project.reset()
        if self.project_id == len(self.projects) - 1:
            self.project_id = 0
        else:
            self.project_id += 1
        self.project = self.projects[self.project_id]

    def act(self):
        """Taking actions is done in an ϵ-greedy fashion to allow sufficient
        exploration: with a chance of ϵ the action is taken randomly and with
        (1-ϵ) the action with the highest value will be executed. ϵ is usually
        decreasing during training to converge the policy's decision making
        process.
        """
        state, durations = self.project.state()
        actions = self.project.get_actions()
        if len(actions) > 1:
            if np.random.rand() < self.epsilon:
                best_action = np.random.choice(actions)
                state_action = self.input_vector(state, best_action)
                value = self.model.predict(state_action)
            else:
                state_action = self.input_vector(state, None)
                value = self.model.predict(state_action)
                best_action = []

                for action in actions:
                    state_action = self.input_vector(state, action)
                    v_temp = self.model.predict(state_action)
                    if v_temp > value:
                        value = v_temp
                        best_action = action
        else:  # there is only the "do-nothing/wait" action
            state_action = self.input_vector(state, None)
            value = self.model.predict(state_action)
            best_action = []
            durations = {}

        return value, best_action, durations, state_action

    def input_vector(self, state, action, topology=True):
        """Helper function to construct the input vector.

        :param state:
        :param action:
        :param topology: when set to true the topology is used when creating
        the input vector
        :return:
        """
        actions = np.zeros((self.num_of_tasks, 1))
        if action:
            actions[action] = 1

        if topology:
            return np.concatenate((actions, state, self.topology)).reshape((1, -1))
        else:
            return np.concatenate((actions, state)).reshape((1, -1))
