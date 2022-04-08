import numpy as np


class Result(object):
    """
    Result accumulators.
    """

    def __init__(self, nb_arms, horizon):
        """
        Create Result Array
        """
        # Store all the choices.
        self.choices = np.zeros(horizon, dtype=int)

        # Store all the rewards, to compute the mean.
        self.rewards = np.zeros(horizon)

        # Store the pulls.
        self.pulls = np.zeros(nb_arms, dtype=int)

    def store(self, time, choice, reward):
        """
        Store results.
        """
        self.choices[time] = choice
        self.rewards[time] = reward
        self.pulls[choice] += 1
