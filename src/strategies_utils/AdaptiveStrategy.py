from typing import AnyStr, List

from src.operators_utils.Operator import Operator
from src.strategies_utils.Strategy import Strategy


class AdaptiveStrategy(Strategy):
    def __init__(self, short_name: AnyStr, arms: List['Operator']):
        super(AdaptiveStrategy, self).__init__(short_name, arms)

        self.arms_num = len(arms)

        self.probabilities = []
        self.rewards = []
        self.losses = []
        self.t = 0

    def setup(self):
        super().setup()

        self.probabilities = [1 / self.arms_num for _ in range(self.arms_num)]
        self.rewards = [0.1 for _ in range(self.arms_num)]
        self.losses = [0.1 for _ in range(self.arms_num)]
        self.t = 0

    def choose_operator_id(self):
        raise NotImplementedError("This method has to be implemented in the class inheriting from AdaptiveStrategy.")

    def update_properties(self, arm_id, reward, loss):
        self.t += 1
        self.rewards[arm_id] = max(0.1, self.rewards[arm_id] + reward)
        self.losses[arm_id] = max(0.1, self.losses[arm_id] + loss)
