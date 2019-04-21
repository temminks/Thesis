from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Sequential


class InceptionV1:
    pass


class Dense128_128:
    def __init__(self, num_of_inputs, model_name, keep_prob=0.8, activation='relu',
                 kernel_initializer='he_normal'):

        self.model = model = Sequential()
        model.add(Dense(128,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                        input_shape=(num_of_inputs, )
                        ))
        model.add(Dropout(rate=1-keep_prob))
        model.add(Dense(128,
                        kernel_initializer=kernel_initializer,
                        activation=activation))
        model.add(Dropout(rate=1-keep_prob))
        model.add(Dense(1,
                        kernel_initializer=kernel_initializer,
                        activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='nadam')

        self.tensorboard = TensorBoard(log_dir="D:/Gregor/Studium/Master 04 - WS 18-19/logs/" +
                                               model_name, write_graph=False, update_freq=100)
