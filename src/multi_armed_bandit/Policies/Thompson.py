r"""
The Thompson Sampling random policy.
"""

import numpy as np

from scipy.stats import beta

try:
    from BasePolicy import BasePolicy
except ImportError:
    from .BasePolicy import BasePolicy


class Thompson(BasePolicy):
    r"""
    The Thompson Sampling random policy.
    """

    def __init__(self, nb_arms, lower=0., amplitude=1.):
        super(Thompson, self).__init__(nb_arms, lower=lower, amplitude=amplitude)

    def __str__(self):
        return r"Thompson"

    def choice(self):
        """
        On each round performs random draw for all arms based on their params (a,b) and
        returns index of arm with the highest draw
        """

        beta_params = zip(self.rewards, self.losses)
        # print(self.rewards, self.losses)
        all_draws = [beta.rvs(i[0], i[1], size=1) for i in beta_params]
        # return np.argmax(all_draws)
        return all_draws.index(max(all_draws))
