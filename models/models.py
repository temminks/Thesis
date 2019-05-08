import keras as K
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Convolution2D, BatchNormalization, Dropout, Dense


class Model:
    def __init__(self, model_name=None):
        self.model_name = model_name if model_name else self.generate_model_name()

        log_dir = "D:/Gregor/Studium/Master 04 - WS 18-19/logs/" + model_name
        self.tensorboard = TensorBoard(log_dir=log_dir, write_graph=False, update_freq=100)

    @staticmethod
    def generate_model_name():
        """Generate a random model name.

        :return: random model name
        """
        return "".join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 7))


class InceptionV1(Model):
    """
    https://github.com/titu1994/Inception-v4/blob/master/inception_resnet_v1.py
    """
    def __init__(self, state_shape, topology_shape, model_name=None):
        super().__init__(model_name)
        input = None

        topology = None
        conv_1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(topology)

        conv_2 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(topology)
        conv_2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv_2)

        conv_3 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(topology)
        conv_3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv_3)
        conv_3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv_3)

        merge = K.layers.merge([conv_1, conv_2, conv_3], concat_axis=1, mode='concat')

        conv = Convolution2D(256, 1, 1, activation='linear', border_mode='same')(merge)

        out = K.layers.merge([conv, input], concat_axis=1, mode='concat')
        out = BatchNormalization(axis=1)(out)

        out = Dense(256, kernel_initializer='he_normal', activation="relu", name='input')(out)
        out = Dense(256, kernel_initializer='he_normal', activation="relu", name='input')(out)
        out = Dense(256, kernel_initializer='he_normal', activation="relu", name='input')(out)

        self.model = Dense(1, kernel_initializer='he_normal', activation="linear",
                           name='input')(out)
        # K.models.Model()


class Dense256_256_256(Model):
    def __init__(self, num_of_inputs, model_name=None, keep_prob=0.8):
        super().__init__(model_name)

        self.model = model = K.models.Sequential([
            Dense(256, kernel_initializer='he_normal', activation="relu",
                  input_dim=num_of_inputs, name='input'),
            Dropout(rate=1 - keep_prob),
            Dense(256, kernel_initializer='he_normal', activation="relu"),
            Dropout(rate=1 - keep_prob),
            Dense(256, kernel_initializer='he_normal', activation="relu"),
            Dropout(rate=1 - keep_prob),
            Dense(1, kernel_initializer='he_normal', activation='linear',
                  name='output')
        ])
        model.compile(loss='mean_squared_error',
                      optimizer='adam')


class Dense256_256_256_256(Model):
    def __init__(self, num_of_inputs, model_name=None, keep_prob=0.8, activation='relu',
                 kernel_initializer='he_normal'):
        super().__init__(model_name)

        self.model = model = K.models.Sequential([
            K.layers.Dense(256,
                           kernel_initializer=kernel_initializer,
                           activation=activation,
                           input_dim=num_of_inputs,
                           name='input'),
            K.layers.Dropout(rate=1 - keep_prob),
            K.layers.Dense(256,
                           kernel_initializer=kernel_initializer,
                           activation=activation),
            K.layers.Dropout(rate=1 - keep_prob),
            K.layers.Dense(256,
                           kernel_initializer=kernel_initializer,
                           activation=activation),
            K.layers.Dropout(rate=1 - keep_prob),
            K.layers.Dense(256,
                           kernel_initializer=kernel_initializer,
                           activation=activation),
            K.layers.Dropout(rate=1 - keep_prob),
            K.layers.Dense(1,
                           kernel_initializer=kernel_initializer,
                           activation='linear',
                           name='output')
        ])
        model.compile(loss='mean_squared_error',
                      optimizer='adam')


class Dense128_128(Model):
    def __init__(self, num_of_inputs, model_name=None, keep_prob=0.8, activation='relu',
                 kernel_initializer='he_normal'):

        super().__init__(model_name)

        self.model = model = K.models.Sequential([
            K.layers.Dense(128,
                           kernel_initializer=kernel_initializer,
                           activation=activation,
                           input_dim=num_of_inputs,
                           name='input'),
            K.layers.Dropout(rate=1-keep_prob),
            K.layers.Dense(128,
                           kernel_initializer=kernel_initializer,
                           activation=activation),
            K.layers.Dropout(rate=1 - keep_prob),
            K.layers.Dense(1,
                           kernel_initializer=kernel_initializer,
                           activation='linear',
                           name='output')
        ])
        model.compile(loss='mean_squared_error',
                      optimizer='adam')
