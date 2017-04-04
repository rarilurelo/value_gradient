# This code is taken from rllab
import numpy as np
import numpy.random as nr


class OUStrategy(object):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, env_spec, mu=0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_space = env_spec.get('action_space')
        self.state = np.ones(self.action_space.shape[0]) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_space.shape[0]) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.action_space.low, self.action_space.high)

