""" Base class for any policy. - If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg,
if rewards are in [-3, 3], lower = -3, amplitude = 6. """
import numpy as np


class BasePolicy(object):
    """ Base class for any policy."""
    def __init__(self, nb_arms, lower=0., amplitude=1.):
        """ New policy."""
        # Parameters
        assert nb_arms > 0, "Error: the 'nb_arms' parameter of a {} object cannot be <= 0.".format(self)

        # Number of arms
        self.nbArms = nb_arms

        # Lower values for rewards
        self.lower = lower
        assert amplitude > 0, "Error: the 'amplitude' parameter of a {} object cannot be <= 0.".format(self)

        # Larger values for rewards
        self.amplitude = amplitude

        # Internal time
        self.t = 0

        # Number of pulls of each arms
        self.pulls = np.zeros(nb_arms, dtype=int)

        # Cumulated rewards of each arms
        self.rewards = np.zeros(nb_arms)

    def __str__(self):
        """ -> str """
        return self.__class__.__name__

    def start_game(self):
        """ Start the game (fill pulls and rewards with 0)."""
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)

    def get_reward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0,
        1]). """
        self.t += 1
        self.pulls[arm] += 1
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] = max(0, self.rewards[arm] + reward)

    def choice(self):
        """ Not defined."""
        raise NotImplementedError(
            "This method choice() has to be implemented in the child class inheriting from BasePolicy.")
