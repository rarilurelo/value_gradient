import tensorflow as tf
import numpy as np
from keras.layers.merge import Concatenate
from keras.layers import Input
import keras.backend as K


class SVG(object):
    def __init__(self, epoch, net, env_spec, replay_buffer, batch_size=32, min_buffer_size=10000, max_path_length=200,
            pi_opimizer=None, q_optimizer=None, pi_lr=1e-3, q_lr=1e-4, sess=None, tau=0.001, gamma=0.99, scale_reward=1.0,
            env=None, noise_fn=None, num_episodes_per_epoch=1, num_critic_update=1, num_policy_update=1, num_update_per_sample=1):
        self.epoch = epoch
        self.net = net
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.max_path_length = max_path_length
        self.in_dim = env_spec.get('observation_space').shape[0]
        self.action_dim = env_spec.get('action_space').shape[0]
        if pi_opimizer is None:
            self.pi_opimizer = tf.train.AdamOptimizer(pi_lr)
        if q_optimizer is None:
            self.q_optimizer = tf.train.AdamOptimizer(q_lr)
        self.sess = sess
        self.tau = tau
        self.gamma = gamma
        self.scale_reward = scale_reward
        self.env = env
        if noise_fn is None:
            self.noise_fn = lambda : np.random.normal(0, 1, size=(self.action_dim, ))
        elif callable(noise_fn):
            self.noise_fn = noise_fn
        elif isinstance(noise_fn, float):
            self.noise_fn = lambda : np.random.normal(0, noise_fn, size=(self.action_dim, ))
        else:
            assert False, 'what do you think about this type of argument? flaot or function or None'
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_critic_update = num_critic_update
        self.num_policy_update = num_policy_update
        self.num_update_per_sample = num_update_per_sample
        self.built = False

    def build(self):
        model = self.net.model
        mu_model = self.net.mu_model
        log_std_model = self.net.log_std_model
        q_model = self.net.q_model
        target_model = self.net.target_model
        target_mu_model = self.net.target_mu_model
        target_log_std_model = self.net.target_log_std_model
        target_q_model = self.net.target_q_model

        self.states = tf.placeholder(tf.float32, shape=(None, self.in_dim), name='states')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')
        self.next_states = tf.placeholder(tf.float32, shape=[None, self.in_dim], name='next_states')
        self.ys = tf.placeholder(tf.float32, shape=[None])

        # There are other implementations about how can we take aciton.
        # Taking next action version or using only mu version or searching action which maximize Q.
        target_mu = target_mu_model(self.states)
        target_log_std = target_log_std_model(self.states)
        target_action = target_mu + K.random_normal(K.shape(target_mu), dtype=tf.float32) * K.exp(target_log_std)
        self.target_q = K.sum(target_q_model(Concatenate()([target_model(self.states), target_action])), axis=-1)

        self.q = K.sum(q_model(Concatenate()([model(self.states), self.actions])), axis=-1)
        self.q_loss = K.mean(K.square(self.ys-self.q))

        self.mu = mu_model(self.states)
        self.log_std = log_std_model(self.states)
        self.eta = (self.actions - self.mu) / K.exp(self.log_std)
        inferred_action = self.mu + K.stop_gradient(self.eta) * K.exp(self.log_std)
        self.pi_loss = - K.mean(q_model(Concatenate()([model(self.states), inferred_action])))

        self.q_updater = self.q_optimizer.minimize(self.q_loss, var_list=self.net.var_q)
        self.pi_updater = self.pi_opimizer.minimize(self.pi_loss, var_list=self.net.var_pi)

        self.soft_updater = [K.update(t_p, t_p*(1-self.tau)+p*self.tau) for p, t_p in zip(self.net.var_all, self.net.var_target_all)]
        self.sync = [K.update(t_p, p) for p, t_p in zip(self.net.var_all, self.net.var_target_all)]

        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def optimize_q(self, batch):
        if not self.built:
            self.build()
        next_q = self.sess.run(self.target_q, {self.states: batch['next_states'], K.learning_phase(): 1})
        ys = batch['rewards'] + (1 - batch['terminals']) * next_q
        feed_in = {
                self.states: batch['states'],
                self.actions: batch['actions'],
                self.rewards: batch['rewards'],
                self.next_states: batch['next_states'],
                self.ys: ys,
                K.learning_phase(): 1
                }
        self.sess.run(self.q_updater, feed_in)

    def optimize_pi(self, batch):
        if not self.built:
            self.build()
        feed_in = {
                self.states: batch['states'],
                self.actions: batch['actions'],
                K.learning_phase(): 1
                }
        self.sess.run(self.pi_updater, feed_in)

    def rollout(self, max_path_length=None):
        if max_path_length is None:
            max_path_length = self.env.spec.tags.get(
                    'wrapper_config.TimeLimit.max_episode_steps')
        states = []
        actions = []
        rewards = []
        terminals = []
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
            terminals.append(d)
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
            terminals=np.array(terminals),
            agent_infos=np.array(agent_infos),
            env_infos=np.array(env_infos),
        )

    def get_action(self, state):
        mu = self.sess.run(self.mu, {self.states: [state], K.learning_phase(): 0})[0]
        log_std = self.sess.run(self.log_std, {self.states: [state], K.learning_phase(): 0})[0]
        a = mu + np.random.normal(size=mu.shape) * np.exp(log_std)
        a = np.clip(a, self.env_spec.get('action_space').low, self.env_spec.get('action_space').high)
        return a, dict(mu=mu, log_std=log_std)

    def train_episode(self):
        if not self.built:
            self.build()
        self.sess.run(self.sync)
        for i in range(self.epoch):
            for _ in range(self.num_episodes_per_epoch):
                episode = self.rollout()
                print('R:', sum(episode['rewards']))
                episode['rewards'] = episode['rewards'] * self.scale_reward
                self.replay_buffer.add_episode(episode)
            batch = self.replay_buffer.random_batch(self.batch_size)
            for _ in range(self.num_critic_update):
                self.optimize_q(batch)
            for _ in range(self.num_policy_update):
                self.optimize_pi(batch)
            self.sess.run(self.soft_updater)

    def train_step(self):
        if not self.built:
            self.build()
        self.sess.run(self.sync)
        train_start = False
        for i in range(self.epoch):
            s = self.env.reset()
            total_reward = 0
            terminal = False
            max_path_length = self.max_path_length
            for step in range(max_path_length):
                if terminal:
                    break
                a, agent_info = self.get_action(s)
                next_s, r, terminal, env_info = self.env.step(a)
                total_reward += r
                self.replay_buffer.add_sample(s, a, r*self.scale_reward, terminal)
                s = next_s
                if self.replay_buffer.size >= self.min_buffer_size:
                    if not train_start:
                        print('train start')
                        train_start = True
                    for _ in range(self.num_update_per_sample):
                        batch = self.replay_buffer.random_batch(self.batch_size)
                        for q_itr in range(self.num_critic_update):
                            self.optimize_q(batch)
                        for pi_itr in range(self.num_policy_update):
                            self.optimize_pi(batch)
                self.sess.run(self.soft_updater)
            print('Total reward: ', total_reward)


