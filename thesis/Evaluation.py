import numpy as np

from thesis.ForwardSarsaLambda import Agent
from thesis.Project import Project
from thesis.Run import J30Runner


class Evaluator(J30Runner):
    def __init__(self, model_name, model):
        super().__init__(train=False)

        self.model_name = model_name
        self.model = model.model
        self.agent = Agent(self.projects, model)
        self.result = []

    def load_weights(self, number):
        self.model.load_weights('.\\models\\' + self.model_name + '\\' + self.model_name +
                                '-' + str(number) + '.h5')

    def evaluate_project(self, project) -> float:
        t = 0

        while not project.is_finished():
            t += project.next(*self.act(project))

        return t

    def evaluate_project_randomly(self, project) -> float:
        t = 0
        while not project.is_finished():
            t += project.next(*self.act_randomly(project))

        return t

    def evaluate(self, num_of_iterations=100):
        """Evaluates a single project for the number of iterations."""
        durations = {}

        for project in self.projects:
            project_list = np.array([Project(project.path, stochastic=project.stochastic)
                                     for _ in range(num_of_iterations)])
            durations[project.path[-8:]] = np.vectorize(self.evaluate_project,
                                                        otypes=[float])(project_list)

        return durations

    def evaluate_all(self, num_of_models, num_of_iterations=100):
        for num_of_model in range(num_of_models):
            print('evaluating model', num_of_model)
            self.load_weights(num_of_model)
            self.result.append(self.evaluate(num_of_iterations))

    def act(self, project):
        """The action with the highest value is executed.

        This function is different from the act-function during training: If no
        tasks are running, the model cannot choose the wait/void action. This
        prevents infinite loops if the wait/void action for such a state has the
        highest q-value.

        :return: the best action and the durations of the tasks in the action
        """
        state, durations = project.state()
        actions = project.get_actions()

        if len(actions) > 1:
            best_action = self.get_best_action(state, actions, project)
            return best_action, durations
        else:
            best_action = []
            return best_action, durations

    @staticmethod
    def act_randomly(project):
        _, durations = project.state()
        best_action = np.random.choice(project.get_actions())
        return best_action, durations

    def get_best_action(self, state, actions, project):
        inputs = np.squeeze(np.array([self.agent.input_vector(state, action)
                                      for action in actions]))
        action_values = np.squeeze(self.model.predict(inputs, len(inputs)))
        max_val = np.argmax(action_values)
        # the wait/void action must not be the best action if there are no running tasks
        if len(project.running) == 0 and actions[max_val] == []:
            max_val = np.argmax(action_values[1:]) + 1
        return actions[max_val]
