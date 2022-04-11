r"""
The epsilon-greedy random policies, with the naive one and some variants.
- At every time step, a fully uniform random exploration has probability :math:`\varepsilon(t)` to happen, otherwise an exploitation is done on accumulated rewards (not means).
- Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
"""

from random import random, seed
import numpy as np
import numpy.random as rn

try:
    from BasePolicy import BasePolicy
except ImportError:
    from .BasePolicy import BasePolicy


#: Default value for epsilon for :class:`EpsilonGreedy`
EPSILON = 0.1


class EpsilonGreedy(BasePolicy):
    r"""
    The epsilon-greedy random policy.
        - At every time step, a fully uniform random exploration has probability :math:`\varepsilon(t)` to happen, otherwise an exploitation is done on accumulated rewards (not means).
        - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nb_arms, seed, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonGreedy, self).__init__(nb_arms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonGreedy class has to be in [0, 1]."  # DEBUG
        self._epsilon = epsilon
        self.seed = seed

    @property
    def epsilon(self):  # Allow child classes to use time-dependent epsilon coef
        return self._epsilon

    def __str__(self):
        return r"EpsilonGreedy($\varepsilon={:.3g}$)".format(self.epsilon)

    def with_probability(self, epsilon):
        seed(self.seed)
        return random() < epsilon

    def choice(self):
        """
        With a probability of epsilon, explore (uniform choice), otherwise exploit based on just accumulated *rewards* (not empirical mean rewards).
        """
        if self.with_probability(self.epsilon):
            rn.seed(self.seed)
            return rn.randint(0, self.nbArms)
        else:
            return np.argmax(self.rewards)
