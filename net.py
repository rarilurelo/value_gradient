import tensorflow as tf
import numpy as np
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential

class Net(object):
    def __init__(self, env_spec):
        self.env_spec = env_spec
        self.in_dim = env_spec.get('observation_space').shape[0]
        self.action_dim = env_spec.get('action_space').n
        self.model, self.pi_model, self.q_model = self.get_model()
        self.target_model, self.target_pi_model, self.target_q_model = self.get_model()

        self.var_pi = self.model.trainable_weights + \
                self.pi_model.trainable_weights
        self.var_target_pi = self.target_model.trainable_weights + \
                self.target_pi_model.trainable_weights

        self.var_q = self.model.trainable_weights + \
                self.q_model.trainable_weights
        self.var_target_q = self.target_model.trainable_weights + \
                self.target_q_model.trainable_weights

        self.var_all = self.model.trainable_weights + \
                self.pi_model.trainable_weights + \
                self.q_model.trainable_weights
        self.var_target_all = self.target_model.trainable_weights + \
                self.target_pi_model.trainable_weights + \
                self.target_q_model.trainable_weights

    def get_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.in_dim))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        pi_model = Sequential()
        pi_model.add(Dense(50, input_dim=50))
        pi_model.add(BatchNormalization())
        pi_model.add(Activation('relu'))
        pi_model.add(Dense(self.action_dim))

        q_model = Sequential()
        q_model.add(Dense(50, input_dim=self.action_dim+50))
        q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(50))
        q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(50))
        q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(1))

        return model, pi_model, q_model


class H():
    def __init__(self):
        self.n = 2
a = H()
env_spec = dict(observation_space=np.array([4,3,4,5]), action_space=a)
net = Net(env_spec)



