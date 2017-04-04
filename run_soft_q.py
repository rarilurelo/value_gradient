import gym
from gym import wrappers
import tensorflow as tf
import os
import argparse
from rllab.envs.gym_env import GymEnv

from net import StochasticNet
from soft_q import SoftQ
from replay_buffer import SimpleReplayPool

parser = argparse.ArgumentParser()
parser.add_argument('--pi_lr', type=float, default=7e-4)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--log', type=str, default='log_soft_q')
args = parser.parse_args()

#log_dir = args.log
#
#if not os.path.exists(log_dir):
#    os.mkdir(log_dir)
#env = gym.make('Pendulum-v0')
env = GymEnv('Pendulum-v0', record_video=True, log_dir=args.log)
env_spec = dict(
        action_space=env.action_space,
        observation_space=env.observation_space)
#env = wrappers.Monitor(env, log_dir, force=True)

net = StochasticNet(env_spec)

replay_buffer = SimpleReplayPool(1000000, env_spec['observation_space'].shape[0], env_spec['action_space'].shape[0])

config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
sess = tf.Session(config=config)

agent = SoftQ(epoch=100000, net=net, env_spec=env_spec, replay_buffer=replay_buffer, env=env, sess=sess, scale_reward=0.01)
        #min_buffer_size=100, K_pi=40)
agent.train_step()


