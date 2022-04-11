r"""
The epsilon-greedy random policy.
"""

from random import random
import numpy as np

try:
    from BasePolicy import BasePolicy
except ImportError:
    from .BasePolicy import BasePolicy


#: Default value for epsilon
EPSILON = 0.1


class EpsilonGreedy(BasePolicy):
    r"""
    The epsilon-greedy random policy.
    """

    def __init__(self, nb_arms, epsilon=EPSILON, lower=0., amplitude=1.):
        super(EpsilonGreedy, self).__init__(nb_arms, lower=lower, amplitude=amplitude)
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for EpsilonGreedy class has to be in [0, 1]."
        self._epsilon = epsilon

    @property
    def epsilon(self):  # Allow child classes to use time-dependent epsilon coef
        return self._epsilon

    def __str__(self):
        return r"EpsilonGreedy($\varepsilon={:.3g}$)".format(self.epsilon)

    def with_probability(self, epsilon):
        return random() < epsilon

    def choice(self):
        """
         At each round the algorithm selects the arm with the highest empirical mean with probability 1 − eps,
         and a random arm with probability eps
        """

        if self.with_probability(self.epsilon):
            return np.random.randint(0, self.nbArms)
        else:
            return np.argmax(self.rewards)
