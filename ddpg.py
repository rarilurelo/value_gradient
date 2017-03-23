import tensorflow as tf
import numpy as np
from keras.layers.merge import Concatenate
from keras.layers import Input
import keras.backend as K

from net import Net

class DDPG(object):
    def __init__(self, net, env_spec,
            pi_opimizer=None, q_optimizer=None, q_optimizer=None, pi_lr=7e-4, q_rate=2.0, sess=None, tau=0.01, gamma=0.99,
            env=None, noise_fn=None):
        self.net = net
        self.env_spec = env_spec
        self.in_dim = env_spec.get('observation_space').shape[0]
        self.action_dim = env_spec.get('action_space').n
        if pi_opimizer is None:
            self.pi_opimizer = tf.train.AdamOptimizer(pi_lr)
        if q_optimizer is None:
            self.q_optimizer = tf.train.AdamOptimizer(pi_lr * q_rate)
        self.sess = sess
        self.tau = tau
        self.gamma = gamma
        self.env = env
        if noise_fn is None:
            self.noise_fn = lambda : np.random.normal(0, 1, size=(self.action_dim, ))
        elif callable(noise_fn):
            self.noise_fn = noise_fn
        elif isinstance(noise_fn, float):
            self.noise_fn = lambda : np.random.normal(0, noise_fn, size=(self,action_dim, ))
        else:
            assert False, 'what do you think? flaot or function or None'
        self.built = False

    def build(self):
        model = self.net.model
        pi_model = self.net.pi_model
        q_model = self.net.q_model
        target_model = self.net.target_model
        target_pi_model = self.net.target_pi_model
        target_q_model = self.net.target_q_model
        self.state = tf.placeholder(tf.float32, shape=(None, self.in_dim), name='state')
        self.action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='action')
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.next_state = tf.placeholder(tf.float32, shape=[None, self.in_dim], name='next_state')
        # terminal contain only 0 or 1 it will work as masking
        self.terminal = tf.placeholder(tf.float32, shape=[None], name='terminal')

        y = self.reward + self.gamma * K.stop_gradient(K.sum(target_q_model(Concatenate()([target_model(self.next_state),
            target_pi_model(target_model(self.next_state))])), axis=-1)) * self.terminal
        self.q = K.sum(q_model(Concatenate()([model(self.state), self.action])), axis=-1)
        self.q_loss = K.mean(K.square(y-self.q))

        self.mu = pi_model(model(self.state))
        self.pi_loss = - K.mean(q_model(Concatenate()([K.stop_gradient(model(self.state)),
            self.mu])))

        self.q_updater = self.q_optimizer.minimize(self.q_loss, var_list=self.net.var_q)
        self.pi_updater = self.pi_opimizer.minimize(self.pi_loss, var_list=self.net.var_pi)

        self.soft_updater = [K.update(t_p, t_p*(1-tau)+p*tau) for p, t_p in zip(self.net.var_all, self.net.var_target_all)]
        self.sync = [K.update(t_p, p) for p, t_p in zip(self.net.var_all, self.var_target_all)]

        self.sess.run(tf.initialize_all_variables())
        self.built = True

    def optimize_q(self, batch):
        if not self.built:
            self.build()
        feed_in = {self.state: batch['state'],
                self.action: batch['action'],
                self.reward: batch['reward'],
                self.next_state: batch['next_state'],
                self.terminal: batch['terminal']}
        self.sess.run(self.q_updater, feed_in)

    def optimize_pi(self, batch):
        if not self.built:
            self.build()
        feed_in = {self.state: batch['state']}
        self.sess.run(self.pi_updater, feed_in)

    def rollout(self, max_path_length=None):
        if max_path_length is None:
            max_path_length = self.env.spec.tags.get(
                    'wrapper_config.TimeLimit.max_episode_steps')
        states = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        s = self.env.reset()
        path_length = 0
        while path_length < max_path_length:
            a, agent_info = self.get_action(s)
            next_s, r, d, env_info = self.env.step(a)
            states.append(s)
            rewards.append(r)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            s = next_s
        return dict(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=np.array(agent_infos),
            env_infos=np.array(env_infos),
        )

    def get_action(self, state):
        mu = self.sess.run(self.mu, {self.state: state})
        a = mu + self.noise_fn()
        return a, dict(mu=mu)

    def train(self):

class H():
    def __init__(self):
        self.n = 2
a = H()
env_spec = dict(observation_space=np.array([4,3,4,5]), action_space=a)
net = Net(env_spec)
ddpg = DDPG(net, env_spec)
ddpg.build()
