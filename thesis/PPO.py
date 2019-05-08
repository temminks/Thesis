# Implementation of the Proximal Policy Optimization algorithm.
#
# Reference to some implementations (most for OpenAI Gym)
# https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
# https://github.com/openai/baselines/tree/master/baselines/ppo2
# http://blog.varunajayasiri.com/ml/ppo.html
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py
# https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py
# https://github.com/tensorforce/tensorforce/blob/master/tensorforce/agents/ppo_agent.py


import keras as K
import numpy as np
from keras.layers import Dense

from thesis.ForwardSarsaLambda import Agent


class PPOBuffer:
    """Trajectories are stored in the buffer."""

    def __init__(self):
        self.memory = dict()


class PPO(Agent):
    """Proximal Policy Optimisation algorithm."""

    def __init__(self, model, policy_size, loss_clipping=0.2):
        self.loss_clipping = loss_clipping
        self.policy_size = policy_size
        self.model = model.model
        self.eps = 1e-10
        self.actor = self.create_policy_head()
        self.critic = self.create_value_head()

    def train(self, num_of_episodes):
        for episode in num_of_episodes:
            state = None
            while self.is_valid_action(self.actor.predict(state)):
                pass

    def is_valid_action(self):
        pass

    def calculate_advantage(self):
        pass

    def create_policy_head(self):
        """The policy head returns a distribution of actions.

        :return:
        """
        return Dense(self.policy_size, kernel_initializer='he_normal', activation="linear",
                     name='policy_output')(self.model)

    def create_value_head(self):
        """The value head predicts the value of a state.

        :return:
        """
        return Dense(1, kernel_initializer='he_normal', activation="linear",
                     name='value_output')(self.model)

    def input_vector(self, state, topology=True):
        """Helper function to construct the input vector.

        :param state:
        :param topology: when set to true the topology is used when creating
        the input vector
        :return:
        """
        if topology:
            return np.concatenate((state, self.topology)).reshape((1, -1))
        else:
            return state.reshape((1, -1))

    def get_action(self, state, topology):
        self.actor.predict(self.input_vector(state, topology))

    def loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + self.eps)
            return -K.mean(K.minimum(r * advantage,
                                     K.clip(r,
                                            min_value=1 - self.loss_clipping,
                                            max_value=1 + self.loss_clipping) * advantage) + ENTROPY_LOSS * (
                                       prob * K.log(prob + self.eps)))

        return loss


class GAE:
    """Generalised Advantage Estimator"""

    def __init__(self):
        pass
