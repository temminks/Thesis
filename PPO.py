# Implementation of the Proximal Policy Optimization algorithm.
#
# Reference to some implementations (most for OpenAI Gym)
# https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
# https://github.com/openai/baselines/tree/master/baselines/ppo2
# http://blog.varunajayasiri.com/ml/ppo.html
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py
# https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py
# https://github.com/tensorforce/tensorforce/blob/master/tensorforce/agents/ppo_agent.py


from collections import namedtuple, deque
from typing import List

import keras as K
import numpy as np

from agents import Agent
from models import models
from project import Project

Trajectory = namedtuple('Trajectory', ['state', 'action', 'reward', 'next_state', 'prob'])


class PPO(Agent):
    """Proximal Policy Optimisation algorithm."""

    def __init__(self, episodes, projects: List[Project], model: models.Model,
                 action_dim, loss_clipping=0.2):
        """

        :param episodes:
        :param projects:
        :param model:
        :param action_dim:
        :param loss_clipping: clipping parameter should be between 0.1 and 0.3
        """
        super().__init__(projects=projects, model=model)

        self.loss_clipping = loss_clipping
        self.action_dim = action_dim
        self.m: models.PPO = model
        self.model: K.Model = model.model
        self.episodes = episodes
        self.buffer = deque()
        self.num_of_trajectories = 10000

        advantage = K.backend.placeholder(shape=(1,))
        old_prediction = K.backend.placeholder(shape=(self.m.action_dim,))

        self.model.compile(optimizer='adam', loss=Losses.gae(advantage, old_prediction))

    def train(self):
        for episode in range(self.episodes):

            for i in range(self.num_of_trajectories):
                self.buffer.append(self._collect_trajectory())

    def _collect_trajectories(self, topology=True):
        """Run a project from starting state 'state' to end.

        :return:
        """
        trajectory = []
        t = 0
        finished = False

        while not finished:
            next_action, durations, value, policy = self._collect_tasks(self.project, topology)
            t += self.project.next(next_action, durations)
            state, durations = self.project.state()
            next_state = self.input_vector(state, next_action, topology)

            if self.project.is_finished():
                trajectory += Trajectory(state=state,
                                         action=next_action,
                                         reward=1 / t,
                                         next_state=next_state,
                                         prob=policy, )
                return trajectory
            else:
                trajectory += Trajectory(state=state,
                                         action=next_action,
                                         reward=0,
                                         next_state=next_state,
                                         prob=policy, )

    def _collect_tasks(self, project, topology=True):
        """Collect tasks and combine them into one action.

        Tasks are collected until either an infeasible tasks is selected by the
        model (i.e. unfulfilled precedence relation) or an additional task
        consumes more resources than available.

        :param topology:
        :return:
        """
        action = []
        possible_actions = project.get_actions()
        state, durations = project.state()

        while True:
            value, policy = self.model.predict(input_vector, topology, np.array(1.1))
            next_task = np.random.choice(a=range(0, self.action_dim), p=policy)

            if [next_task] + action in possible_actions():
                action = [next_task] + action
                project.next(action, durations)
                state, durations = project.state()
                input_vector = self.input_vector(state, action, topology)
            else:
                return action, durations, value, policy


class Losses:
    @staticmethod
    def gae(advantage, old_prediction,
            loss_clipping=0.2, value_coeff=0.5, entropy_coeff=0.01):
        def loss(y_true, y_pred):
            """Keras uses a loss function with two arguments: y_true and y_pred.

            :param y_true:
            :param y_pred:
            :return:
            """

            print(y_true, y_pred)

            prob = 0
            old_prob = y_true * old_prediction
            ratio = prob / (old_prob + 1e-10)

            return - Losses._clipped_loss(ratio, advantage, loss_clipping) \
                   - value_coeff * Losses._value_loss(y_true, y_pred) \
                   + entropy_coeff * Losses._entropy_bonus(prob)
        return loss

    @staticmethod
    def _clipped_loss(ratio, advantage, loss_clipping):
        return K.backend.mean(K.backend.minimum(ratio * advantage,
                                                K.backend.clip(ratio,
                                                               min_value=1 - loss_clipping,
                                                               max_value=1 + loss_clipping) * advantage))

    @staticmethod
    def _value_loss(v_true, v_pred):
        """Mean squared error for the value function."""
        return K.backend.mean(K.backend.square(v_pred - v_true))

    @staticmethod
    def _entropy_bonus(prob):
        return K.backend.mean(prob * K.backend.log(prob + 1e-10))
