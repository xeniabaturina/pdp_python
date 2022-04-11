r"""
The UCB1 random policy.
"""

import numpy as np

try:
    from BasePolicy import BasePolicy
except ImportError:
    from .BasePolicy import BasePolicy


class UCB1(BasePolicy):
    r"""
    The UCB1 random policy.
    """

    def __init__(self, nb_arms, lower=0., amplitude=1.):
        super(UCB1, self).__init__(nb_arms, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"UCB1"

    def choice(self):
        """
        Initially, each arm is played once. Afterwards, at round t, the algorithm greedily picks the arm according to
        the number of times that each arm has been played in addition to the empirical means
        """

        if self.t < 5:
            return self.t

        return np.argmax([
            mu_i + np.sqrt(2 * np.log(self.t) / self.pulls[i])
            for i, mu_i in enumerate(self.rewards)
        ])
