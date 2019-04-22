import keras as K
from keras.callbacks import TensorBoard


class InceptionV1:
    pass


class Dense128_128:
    def __init__(self, num_of_inputs, model_name, keep_prob=0.8, activation='relu',
                 kernel_initializer='he_normal'):
        self.model = model = K.models.Sequential([
            K.layers.Dense(128,
                           kernel_initializer=kernel_initializer,
                           activation=activation,
                           input_shape=(num_of_inputs,)),
            K.layers.Dropout(rate=1-keep_prob),
            K.layers.Dense(128,
                           kernel_initializer=kernel_initializer,
                           activation=activation),
            K.layers.Dropout(rate=1-keep_prob),
            K.layers.Dense(1,
                           kernel_initializer=kernel_initializer,
                           activation='linear')
        ])
        model.compile(loss='mean_squared_error',
                      optimizer='nadam')

        log_dir = "D:/Gregor/Studium/Master 04 - WS 18-19/logs/" + model_name
        self.tensorboard = TensorBoard(log_dir=log_dir,
                                       write_graph=False,
                                       update_freq=100)
