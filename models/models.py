from abc import ABCMeta, abstractmethod

import keras as K
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, BatchNormalization, Dropout, Dense, Input, concatenate, Flatten


class Model(metaclass=ABCMeta):
    def __init__(self, model_name=None):
        self.model_name = model_name if model_name else self.generate_model_name()

        log_dir = "D:/Gregor/Studium/Master 04 - WS 18-19/logs/" + self.model_name
        self.tensorboard = TensorBoard(log_dir=log_dir, write_graph=False, update_freq=100)

    @property
    @abstractmethod
    def model(self):
        """Needs to be implemented by each child class."""
        pass

    @staticmethod
    def generate_model_name() -> str:
        """Generate a random model name."""
        return "".join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 7))


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
    """Input -> FC-RELU -> DROPOUT -> FC-RELU -> DROPOUT -> FC(1)"""

    def __init__(self, num_of_inputs, model_name=None, keep_prob=0.8):
        super().__init__(model_name)

        self.model = model = K.models.Model([
            K.layers.Dense(128, kernel_initializer='he_normal', activation='relu',
                           input_dim=num_of_inputs, name='input'),
            K.layers.Dropout(rate=1 - keep_prob),
            K.layers.Dense(128, kernel_initializer='he_normal', activation='relu'),
            K.layers.Dropout(rate=1 - keep_prob),
            K.layers.Dense(1, activation='linear', name='output')
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')


class PPO(Model):
    """

    https://github.com/titu1994/Inception-v4/blob/master/inception_resnet_v1.py"""

    def __init__(self, state_dim, action_dim, model_name=None, keep_prob=0.8):
        super().__init__(model_name)

        self.state = Input(shape=(state_dim,), name='State_Input')
        self.topology = Input(shape=(32, 32, 1), name='Topology_Input')
        self.advantage = Input(shape=(1,), name='Advantage_Input')
        self.action_dim = action_dim

        self.model: Model = self.build_model(self.state, self.topology, self.advantage,
                                             keep_prob, self.action_dim)

    @staticmethod
    def build_model(state: Input,
                    topology: Input,
                    advantage: Input,
                    keep_prob: float,
                    action_dim, ) -> Model:
        # Input -> FC(128)-RELU -> DROPOUT -> FC(128)-RELU -> DROPOUT
        sx = Dense(128, kernel_initializer='he_normal', activation='relu',
                   name='input')(state)
        sx = Dropout(rate=1 - keep_prob)(sx)
        sx = Dense(128, kernel_initializer='he_normal', activation='relu')(sx)
        sx = Dropout(rate=1 - keep_prob)(sx)

        # Input -> CONV / CONV->CONV / CONV->CONV->CONV ->CONCAT -> NORM -> CONV
        conv_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(topology)

        conv_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(topology)
        conv_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv_2)

        conv_3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(topology)
        conv_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv_3)
        conv_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv_3)

        concat = K.layers.concatenate([conv_1, conv_2, conv_3])
        concat = BatchNormalization()(concat)

        conv = Conv2D(filters=256, kernel_size=(1, 1), padding='same')(concat)
        conv = Flatten()(conv)

        # CONCAT -> FC(256) -> FC(256) -> Value-Head / Policy-Head
        out = concatenate([sx, conv])
        out = Dense(256, kernel_initializer='he_normal', activation='relu')(out)
        out = Dropout(rate=1 - keep_prob)(out)
        out = Dense(256, kernel_initializer='he_normal', activation='relu')(out)
        out = Dropout(rate=1 - keep_prob)(out)

        value = Dense(1, kernel_initializer='he_normal', name='critic')(out)
        policy = Dense(action_dim, kernel_initializer='he_normal',
                       activation='softmax', name='actor')(out)

        return K.Model(inputs=[state, topology, advantage],
                       outputs=[value, policy],
                       name='resourcenet')
