from typing import AnyStr, List

import numpy as np

from src.graph_utils import Graph
from src.graph_utils.Graph import Mode
from src.operators_utils import Operator


class Strategy(object):
    def __init__(self, short_name: AnyStr, operators: List['Operator']):
        self.short_name = short_name
        self.operators = operators

        self.pulls = []
        self.pulls_succ = []

    def setup(self):
        self.pulls = np.zeros(len(self.operators), dtype=int)
        self.pulls_succ = np.zeros(len(self.operators), dtype=int)

    def choose_operator_id(self):
        raise NotImplementedError("This method has to be implemented in the class inheriting from Strategy.")

    def apply_operator(self, operator_id, g: Graph, calculated_weights):
        self.pulls[operator_id] += 1

        old_weight = g.get_path_weight(calculated_weights)

        g_mutated = self.operators[operator_id].apply_with_log(g, calculated_weights)

        g_new = Graph(Mode.EDGES, g_mutated.edges, g.dataset_dir)
        g_new.set_path_weight(g_mutated.get_path_weight(calculated_weights))
        g_new.coordinates_from_csv = g.coordinates_from_csv

        new_weight = g_new.get_path_weight(calculated_weights)

        reward = (1 - new_weight / old_weight) * 100
        loss = (new_weight / old_weight) * 100

        if reward > 0:
            self.pulls_succ[operator_id] += 1

        self.update_properties(operator_id, reward, loss)

        return g_new

    def update_properties(self, operator_id, reward, loss):
        raise NotImplementedError("This method has to be implemented in the class inheriting from Strategy.")
