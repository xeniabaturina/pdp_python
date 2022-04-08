import numpy as np


class MAB(object):

    def __init__(self, configuration):
        """New MAB."""
        self.arms = []  #: List of arms

        if isinstance(configuration, dict):
            arm_type = configuration["arm_type"]
            params = configuration["params"]
            for param in params:
                self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param))
            self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None
        else:
            for arm in configuration:
                self.arms.append(arm)

        # Means of arms
        self.means = np.array([0 for _ in self.arms])

        # Number of arms
        self.nbArms = len(self.arms)

        # Max mean of arms
        self.maxArm = np.max(self.means)

        # Min mean of arms
        self.minArm = np.min(self.means)

    def __repr__(self):
        return "{}(nbArms: {}, arms: {}, minArm: {:.3g}, maxArm: {:.3g})".format(self.__class__.__name__, self.nbArms,
                                                                                 self.arms, self.minArm, self.maxArm)

    def draw(self, arm_id, v, t=1):
        """ Return a random sample from the armId-th arm, at time t. Usually t is not used."""
        return self.arms[arm_id].draw(v, t)
