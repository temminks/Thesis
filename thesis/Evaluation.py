from keras.layers import Dense
from keras.models import Sequential

from thesis.ForwardSarsaLambda import ForwardSarsaLambda


class Evaluator:
    def __init__(self, project, model_name, model=None):
        self.model = self.build_model()
        self.project = project
        self.num_of_tasks = self.project.num_of_tasks
        self.topology = self.project.adjacency()
        self.model_name = model_name
        if not model:
            self.model = self.build_model()
        else:
            self.model = model

    def load_weights(self):
        self.model.load_weights('.\\models\\m-' + self.model_name + '\\' + self.model_name)

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Dense(128,
                        kernel_initializer='he_normal',
                        activation='relu',
                        input_shape=(1266, )))
        model.add(Dense(128,
                        kernel_initializer='he_normal',
                        activation='relu'))
        model.add(Dense(1,
                        kernel_initializer='he_normal',
                        activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def evaluate(self):
        """Runs a single project and returns the duration."""
        t = 0

        while not self.project.finished():
            t += self.project.next(*self.act())

        self.project.reset()
        return t

    def act(self):
        """The action with the highest value is executed.

        This function is different from the act-function during training: If no
        tasks are running, the model cannot choose the wait/void action. This
        prevents infinite loops if the wait/void action for such a state has the
        highest q-value.

        :return: the best action and the durations of the tasks in the action
        """
        state, durations = self.project.state()
        actions = self.project.get_actions()

        if len(actions) > 1:
            value = -100000000
            for action in actions:
                state_action = ForwardSarsaLambda.input_vector(state, action)
                v_temp = self.model.model.predict(state_action)
                if v_temp > value and len(self.project.running) > 0:
                    value = v_temp
                    best_action = action

            return best_action, durations
        else:
            best_action = []
            return best_action, durations
