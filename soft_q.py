import tensorflow as tf
import numpy as np
from keras.layers.merge import Concatenate
from keras.layers import Input
import keras.backend as K


class SoftQ(object):
    def __init__(self, epoch, net, env_spec, replay_buffer, env, sess, batch_size=64, min_buffer_size=10000,
            max_path_length=200, pi_opimizer=None, q_optimizer=None,
            pi_lr=1e-4, q_lr=1e-3, tau=0.001, gamma=0.99, scale_reward=1.0, K_v=50, K_pi=32, M_pi=32, entropy_alpha=0.1,
            num_episodes_per_epoch=1, num_critic_update=1, num_policy_update=1, num_update_per_sample=1):
        self.epoch = epoch
        self.net = net
        self.env_spec = env_spec
        self.in_dim = env_spec.get('observation_space').shape[0]
        self.action_dim = env_spec.get('action_space').shape[0]
        self.replay_buffer = replay_buffer
        self.env = env
        self.sess = sess
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.max_path_length = max_path_length
        if pi_opimizer is None:
            self.pi_opimizer = tf.train.AdamOptimizer(pi_lr)
        if q_optimizer is None:
            self.q_optimizer = tf.train.AdamOptimizer(q_lr)
        self.tau = tau
        self.gamma = gamma
        self.scale_reward = scale_reward
        self.K_v = K_v
        self.K_pi = K_pi
        self.M_pi = M_pi
        self.entropy_alpha = entropy_alpha
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_critic_update = num_critic_update
        self.num_policy_update = num_policy_update
        self.num_update_per_sample = num_update_per_sample
        # WIP
        self.use_pi = lambda x: False
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

        self.states = tf.placeholder(tf.float32, shape=[None, self.in_dim], name='states')
        self.actions = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')
        self.ys = tf.placeholder(tf.float32, shape=[None])

        # mu: (batch, action)
        self.mu = mu = mu_model(self.states)
        self.log_std = log_std = log_std_model(self.states)

        # estimate V
        shape_for_v = K.concatenate([[self.K_v], K.shape(mu)], axis=0)
        # sampled_action: (sample, batch, action)
        sampled_action_for_v = mu + K.random_normal(shape_for_v, dtype=tf.float32) * K.exp(log_std)
        # prob_density: (sample, batch, 1)
        prob_density = 1/(np.sqrt(2*np.pi)**self.action_dim*K.prod(K.exp(log_std), axis=1, keepdims=True)) * \
                K.exp(-1/2*K.sum((sampled_action_for_v - mu)**2/(K.exp(log_std)**2), axis=2, keepdims=True))
        # Q: (sample, batch, 1)
        Q = target_q_model(Concatenate()([K.permute_dimensions(K.repeat(target_model(self.states), self.K_v), (1, 0, 2)), sampled_action_for_v]))
        # target_v_from_pi: (batch)
        self.target_v_from_pi = K.sum(self.entropy_alpha * K.log(K.mean(K.exp(1/self.entropy_alpha * Q) / prob_density, axis=0)), axis=-1)

        # action space may be -10 ~ +10
        sampled_action_for_v = K.random_uniform(shape_for_v, minval=-10, maxval=10, dtype=tf.float32)
        prob_density = K.ones_like(sampled_action_for_v) / 20
        Q = target_q_model(Concatenate()([K.permute_dimensions(K.repeat(target_model(self.states), self.K_v), (1, 0, 2)), sampled_action_for_v]))
        self.target_v_from_uniform = K.sum(self.entropy_alpha * K.log(K.mean(K.exp(1/self.entropy_alpha * Q) / prob_density, axis=0)), axis=-1)

        self.q = K.sum(q_model(Concatenate()([model(self.states), self.actions])), axis=-1)
        self.q_loss = K.mean(0.5*K.square(self.ys-self.q))

        # pi
        shape_for_M = K.concatenate([[self.M_pi], K.shape(mu)], axis=0)
        shape_for_K = K.concatenate([[self.K_pi], K.shape(mu)], axis=0)
        self.sampled_action_for_M = mu + K.random_normal(shape_for_M, dtype=tf.float32) * K.exp(log_std)
        sampled_action_for_K = mu + K.random_normal(shape_for_K, dtype=tf.float32) * K.exp(log_std)

        self.h = tf.placeholder(tf.float32, shape=[None], name='h')
        self.sampled_action_feeder = tf.placeholder(tf.float32, shape=[self.M_pi, None, self.action_dim], name='sampled_action_feeder')

        # tile_for_K: (M, K, batch, action)
        tile_for_K = K.tile(K.expand_dims(sampled_action_for_K, 0), [self.M_pi, 1, 1, 1])
        # tile_for_M: (K, M, batch, action)
        tile_for_M = K.tile(K.expand_dims(self.sampled_action_feeder, 0), [self.K_pi, 1, 1, 1])
        # kappa: (K, M, batch, 1)
        kappa = K.exp(-1/self.h * K.sum((tile_for_M - K.permute_dimensions(tile_for_K, (1, 0, 2, 3)))**2, axis=3, keepdims=True))
        # grad_kappa: (K, M, batch, action)
        grad_kappa = -2/self.h * (tile_for_M - K.permute_dimensions(tile_for_K, (1, 0, 2, 3))) * \
                K.exp(-1/self.h * K.sum((tile_for_M - K.permute_dimensions(tile_for_K, (1, 0, 2, 3)))**2, axis=3, keepdims=True))
        # _Q: (M, batch, 1)
        _Q = q_model(Concatenate()([K.permute_dimensions(K.repeat(model(self.states), self.M_pi), (1, 0, 2)), self.sampled_action_for_M]))
        # grad_Q: (M, batch, action)
        grad_Q = K.gradients(_Q, self.sampled_action_for_M)[0]
        # delta_f: (K, M, batch, action) -> (K, batch, action)
        delta_f = K.stop_gradient(K.mean(kappa*grad_Q + grad_kappa, axis=1))
        self.pi_loss = K.mean(delta_f * sampled_action_for_K)

        self.q_updater = self.q_optimizer.minimize(self.q_loss, var_list=self.net.var_q)
        self.pi_updater = self.pi_opimizer.minimize(self.pi_loss, var_list=self.net.var_pi)

        self.soft_updater = [K.update(t_p, t_p*(1-self.tau)+p*self.tau) for p, t_p in zip(self.net.var_all, self.net.var_target_all)]
        self.sync = [K.update(t_p, p) for p, t_p in zip(self.net.var_all, self.net.var_target_all)]

        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def optimize_q(self, batch, epoch):
        if not self.built:
            self.build()
        if self.use_pi(epoch):
            next_v = self.sess.run(self.target_v_from_pi, {self.states: batch['next_states'], K.learning_phase(): 1})
        else:
            next_v = self.sess.run(self.target_v_from_uniform, {self.states: batch['next_states'], K.learning_phase(): 1})
        ys = batch['rewards'] + self.gamma * (1 - batch['terminals']) * next_v
        feed_in = {
                self.states: batch['states'],
                self.actions: batch['actions'],
                self.rewards: batch['rewards'],
                self.ys: ys,
                K.learning_phase(): 1
                }
        self.sess.run(self.q_updater, feed_in)

    def optimize_pi(self, batch):
        if not self.built:
            self.build()
        sampled_action_for_M = self.sess.run(self.sampled_action_for_M, {self.states: batch['states']})
        sampled_action = np.transpose(sampled_action_for_M, (1, 0, 2))[:, :, np.newaxis, :]
        pairwise_d = np.sum((np.tile(sampled_action, (self.M_pi, 1)) - \
            np.transpose(np.tile(sampled_action, (self.M_pi, 1)), (0, 2, 1, 3)))**2, axis=3).reshape(sampled_action.shape[0], -1)
        d = np.median(pairwise_d, axis=1)
        h = d/(2*np.log(self.M_pi+1))
        feed_in = {
                self.states: batch['states'],
                self.actions: batch['actions'],
                self.sampled_action_feeder: sampled_action_for_M,
                self.h: h,
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
        step_count = 0
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
                            self.optimize_q(batch, i)
                        for pi_itr in range(self.num_policy_update):
                            self.optimize_pi(batch)
                if step_count%1000 == 0:
                    self.sess.run(self.sync)
                step_count += 1
            print('Total reward: ', total_reward)



