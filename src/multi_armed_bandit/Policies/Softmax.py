r"""
The Softmax random policy.
"""

import numpy as np

try:
    from BasePolicy import BasePolicy
except ImportError:
    from .BasePolicy import BasePolicy


#: Default value for tau
TAU = 10


class Softmax(BasePolicy):
    r"""
    The softmax random policy.
    """

    def __init__(self, nb_arms, tau=TAU, lower=0., amplitude=1.):
        super(Softmax, self).__init__(nb_arms, lower=lower, amplitude=amplitude)
        assert 0 <= tau, "Error: the 'tau' parameter for Softmax class has to be in [0, inf]."
        self._tau = tau

    @property
    def tau(self):  # Allow child classes to use time-dependent tau coef
        return self._tau

    def __str__(self):
        return r"Softmax($\vartau={:.3g}$)".format(self.tau)

    def choice(self):
        """
        Pick each arm with a probability that is proportional to its average reward.
        """

        normalization_factor = np.sum([np.exp(mu_j / self.tau) for mu_j in self.rewards])

        probabilities = np.array([
            np.exp(mu_i / self.tau) / normalization_factor
            for mu_i in self.rewards
        ])

        if np.nan in probabilities:
            print(self.rewards)
            print(normalization_factor)
            print(probabilities)

        arm = np.random.choice(
            a=[0, 1, 2, 3, 4],
            p=probabilities
        )
        return arm
