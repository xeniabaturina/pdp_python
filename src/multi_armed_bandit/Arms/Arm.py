class Arm(object):

    def __init__(self, lower=0., amplitude=1.):
        # Lower value of rewards
        self.lower = lower

        # Amplitude of value of rewards
        self.amplitude = amplitude

        # Lower value of rewards
        self.min = lower

        # Higher value of rewards
        self.max = lower + amplitude

    def draw(self, v, t=None):
        raise NotImplementedError("This method draw has to be implemented in the class inheriting from Arm.")
