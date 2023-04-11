import numpy as np

from src.strategies_utils.Strategy import Strategy


class StatisticalStrategy(Strategy):
    def __init__(self, short_name, operators, probabilities):
        super(StatisticalStrategy, self).__init__(short_name, operators)

        self.operators_num = len(operators)

        self.probabilities = probabilities

    def choose_operator_id(self):
        operator_id = np.random.choice(
            a=self.operators_num,
            p=self.probabilities
        )

        return operator_id

    def update_properties(self, operator_id, reward, loss):
        pass
