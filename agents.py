from typing import List

import numpy as np

from models import models
from project import Project


class Agent:
    def __init__(self, projects: List[Project], model: models.Model):
        """

        :param projects:
        :param model:
        """
        self.model = model.model
        self.projects = projects
        self.project_id = 0
        self.project = projects[self.project_id]
        self.tensorboard = model.tensorboard
        self.model_name = model.model_name
        self.topology = np.reshape(self.project.topology(), (-1, 1))
        self.num_of_tasks = self.project.num_of_tasks

    def next_project(self):
        """Reset the current project and switch the project to the next one."""
        self.project.reset()
        if self.project_id == len(self.projects) - 1:
            self.project_id = 0
        else:
            self.project_id += 1
        self.project = self.projects[self.project_id]
        self.num_of_tasks = self.project.num_of_tasks

    def input_vector(self, state, action, topology=True) -> np.array:
        """Helper function to construct the input vector.

        :param state:
        :param action:
        :param topology: when set to true the topology is used when creating
        the input vector
        :return:
        """
        actions = np.zeros((self.num_of_tasks, 1))
        actions[action] = 1

        if topology:
            return np.concatenate((actions, state, self.topology)).reshape((1, -1))
        else:
            return np.concatenate((actions, state)).reshape((1, -1))

    def get_best_action(self, state, actions):
        """

        :param state:
        :param actions:
        :return:
        """
        inputs = np.squeeze(np.array([self.input_vector(state, action) for action in actions]))
        action_values = np.squeeze(self.model.predict(inputs, len(inputs)))
        max_val = np.argmax(action_values)
        return action_values[max_val], actions[max_val], self.input_vector(state, actions[max_val])
