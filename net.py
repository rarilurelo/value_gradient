import tensorflow as tf
import numpy as np
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential
from keras.initializers import uniform

class DeterministicNet(object):
    def __init__(self, env_spec, h_dim=32):
        self.env_spec = env_spec
        self.h_dim = h_dim
        self.in_dim = env_spec.get('observation_space').shape[0]
        self.action_dim = env_spec.get('action_space').shape[0]
        self.model, self.pi_model, self.q_model = self.get_model()
        self.target_model, self.target_pi_model, self.target_q_model = self.get_model()

        self.var_pi = self.pi_model.trainable_weights
        self.var_target_pi = self.target_pi_model.trainable_weights

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
        model.add(Dense(self.h_dim, input_dim=self.in_dim))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(self.h_dim))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))

        small_uniform = uniform(-3e-3, 3e-3)
        q_model = Sequential()
        q_model.add(Dense(self.h_dim, input_dim=self.action_dim+self.h_dim))
        #q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(self.h_dim))
        #q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(self.h_dim))
        #q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(1, kernel_initializer=small_uniform, bias_initializer=small_uniform))

        pi_model = Sequential()
        pi_model.add(Dense(self.h_dim, input_dim=self.in_dim))
        #pi_model.add(BatchNormalization())
        pi_model.add(Activation('relu'))
        pi_model.add(Dense(self.h_dim))
        #pi_model.add(BatchNormalization())
        pi_model.add(Activation('relu'))
        pi_model.add(Dense(self.action_dim, activation='tanh', kernel_initializer=small_uniform, bias_initializer=small_uniform))

        return model, pi_model, q_model

class StochasticNet(object):
    def __init__(self, env_spec, h_dim=32):
        self.env_spec = env_spec
        self.h_dim = h_dim
        self.in_dim = env_spec.get('observation_space').shape[0]
        self.action_dim = env_spec.get('action_space').shape[0]
        self.model, self.mu_model, self.log_std_model, self.q_model = self.get_model()
        self.target_model, self.target_mu_model, self.target_log_std_model, self.target_q_model = self.get_model()

        s = Input((self.in_dim, ))
        mu = self.mu_model(s)
        log_std = self.log_std_model(s)
        m = Model(inputs=s, outputs=[mu, log_std])
        self.var_pi = m.trainable_weights
        target_mu = self.target_mu_model(s)
        target_log_std = self.target_log_std_model(s)
        target_m = Model(inputs=s, outputs=[target_mu, target_log_std])
        self.var_target_pi = target_m.trainable_weights

        self.var_q = self.model.trainable_weights + \
                self.q_model.trainable_weights
        self.var_target_q = self.target_model.trainable_weights + \
                self.target_q_model.trainable_weights

        self.var_all = self.model.trainable_weights + \
                m.trainable_weights + \
                self.q_model.trainable_weights
        self.var_target_all = self.target_model.trainable_weights + \
                target_m.trainable_weights + \
                self.target_q_model.trainable_weights

    def get_model(self):
        model = Sequential()
        model.add(Dense(self.h_dim, input_dim=self.in_dim))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(self.h_dim))
        #model.add(BatchNormalization())
        model.add(Activation('relu'))

        small_uniform = uniform(-3e-3, 3e-3)

        q_model = Sequential()
        q_model.add(Dense(self.h_dim, input_dim=self.action_dim+self.h_dim))
        #q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(self.h_dim))
        #q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(self.h_dim))
        #q_model.add(BatchNormalization())
        q_model.add(Activation('relu'))
        q_model.add(Dense(1, kernel_initializer=small_uniform, bias_initializer=small_uniform))

        pi_model = Sequential()
        pi_model.add(Dense(self.h_dim, input_dim=self.in_dim))
        #pi_model.add(BatchNormalization())
        pi_model.add(Activation('relu'))
        pi_model.add(Dense(self.h_dim))
        #pi_model.add(BatchNormalization())
        pi_model.add(Activation('relu'))
        pi_model.add(Dense(self.action_dim, activation='tanh', kernel_initializer=small_uniform, bias_initializer=small_uniform))
        mu_model = Sequential([pi_model])
        mu_model.add(Dense(self.action_dim, activation='tanh', kernel_initializer=small_uniform, bias_initializer=small_uniform))
        log_std_model = Sequential([pi_model])
        log_std_model.add(Dense(self.action_dim, activation='tanh', kernel_initializer=small_uniform, bias_initializer=small_uniform))

        return model, mu_model, log_std_model, q_model


