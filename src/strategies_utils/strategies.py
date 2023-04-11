import warnings
import random
from typing import List

import numpy as np
from scipy.stats import beta

from src.operators_utils.Operator import Operator
from src.statistics_utils.Statistics import Statistics
from src.strategies_utils.AdaptiveStrategy import AdaptiveStrategy
from src.strategies_utils.StatisticalStrategy import StatisticalStrategy


class EpsilonGreedy(AdaptiveStrategy):
    def __init__(self, arms: List[Operator], epsilon: float):
        assert 0 <= epsilon <= 1, 'Error: the epsilon parameter for EpsilonGreedy class has to be in [0, 1]'
        super(EpsilonGreedy, self).__init__('eps-greedy-' + str(epsilon), arms)

        self.epsilon = epsilon

    def with_probability_epsilon(self):
        return random.random() < self.epsilon

    def choose_operator_id(self):
        self.probabilities = [self.epsilon for _ in range(self.arms_num)]
        self.probabilities[np.argmax(self.rewards)] = 1 - self.epsilon

        if self.with_probability_epsilon():
            return np.random.randint(0, self.arms_num)
        else:
            return np.argmax(self.rewards)


class UCB1(AdaptiveStrategy):
    def __init__(self, arms: List[Operator]):
        super(UCB1, self).__init__('UCB1', arms)

    def choose_operator_id(self):
        if self.t < self.arms_num:
            return self.t

        arm_id = np.argmax([
            mu_i + np.sqrt(2 * np.log(self.t) / self.pulls[i])
            for i, mu_i in enumerate(self.rewards)
        ])

        self.probabilities = [0 for _ in range(self.arms_num)]
        self.probabilities[arm_id] = 1

        return arm_id


class Softmax(AdaptiveStrategy):
    def __init__(self, arms: List[Operator], tau: float):
        assert tau >= 0, 'Error: the tau parameter for Softmax class has to be in [0, inf]'
        super(Softmax, self).__init__('softmax-' + str(tau), arms)

        self.tau = tau

    def choose_operator_id(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            normalization_factor = np.sum([np.exp(mu_j / self.tau) for mu_j in self.rewards])

            probabilities = np.array([
                np.exp(mu_i / self.tau) / normalization_factor
                for mu_i in self.rewards
            ])

            self.probabilities = list(probabilities)

            arm_id = np.random.choice(
                a=[i for i in range(self.arms_num)],
                p=self.probabilities
            )

            return arm_id


class Thompson(AdaptiveStrategy):
    def __init__(self, arms: List[Operator]):
        super(Thompson, self).__init__('thompson', arms)

    def choose_operator_id(self):
        beta_params = zip(self.rewards, self.losses)

        all_draws = [beta.rvs(i[0], i[1], size=1) for i in beta_params]

        self.probabilities = [0 for _ in range(self.arms_num)]
        self.probabilities[all_draws.index(max(all_draws))] = 1

        return all_draws.index(max(all_draws))


class S_n(StatisticalStrategy):
    def __init__(self, operators: List[Operator]):
        super(S_n, self).__init__('s_n', operators, [0.25, 0.25, 0.25, 0.25])


class Random(StatisticalStrategy):
    def __init__(self, operators: List[Operator]):
        super(Random, self).__init__('random', operators, [1 / len(operators) for _ in operators])


class S_fn(StatisticalStrategy):
    def __init__(self, operators: List[Operator]):
        super(S_fn, self).__init__('s_fn', operators, [0.4435, 0.1233, 0.2152, 0.218])


def normalize(raw):
    return [float(i)/sum(raw) for i in raw]


class S_v(StatisticalStrategy):
    def __init__(self, operators: List[Operator], statistics: Statistics):
        super(S_v, self).__init__('s_v', operators,
                                  normalize([statistics.operator_statistics(operator_id)['v'] /
                                             statistics.operator_statistics(operator_id)['t']
                                             for operator_id, _ in enumerate(operators)]))


class S_f(StatisticalStrategy):
    def __init__(self, operators: List[Operator], statistics: Statistics):
        super(S_f, self).__init__('s_f', operators,
                                  normalize([statistics.operator_statistics(operator_id)['f']
                                             for operator_id, _ in enumerate(operators)]))


class S_pool(StatisticalStrategy):
    def __init__(self, operators: List[Operator], statistics: Statistics):
        super(S_pool, self).__init__('s_pool', operators,
                                     normalize([statistics.operator_statistics(operator_id)['is_in_pool']
                                                for operator_id, _ in enumerate(operators)]))


class S_best(StatisticalStrategy):
    def __init__(self, operators: List[Operator], statistics: Statistics):
        super(S_best, self).__init__('s_best', operators,
                                     normalize([statistics.operator_statistics(operator_id)['was_best']
                                                for operator_id, _ in enumerate(operators)]))


class Pure(StatisticalStrategy):
    def __init__(self, operator: Operator):
        super(Pure, self).__init__('pure-' + str(operator.short_name), [operator], [1])
