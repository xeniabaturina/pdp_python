try:
    from .Arm import Arm
except ImportError:
    from Arm import Arm


class RelocateBlock(Arm):
    """
    RelocateBlock-mutation arm
    """

    def draw(self, v, t=None):
        """ The parameter t is ignored in this Arm."""
        return v
