from time import perf_counter
from typing import AnyStr


from src.graph_utils import Graph
from src.logs_utils import Logger


class Operator(object):
    def __init__(self, short_name: AnyStr, logger: Logger):
        self.short_name = short_name
        self.logger = logger

    def apply_with_log(self, g: Graph, calculated_weights):
        old_weight = g.get_path_weight()
        start_time = perf_counter()

        g_new = self.apply(g, calculated_weights)

        end_time = perf_counter()
        new_weight = g_new.get_path_weight()

        elapsed_time_s = end_time - start_time
        elapsed_time_ms = elapsed_time_s * 1000

        improvement_pct = (1 - new_weight / old_weight) * 100
        is_succ = 1 if improvement_pct > 0 else 0

        self.logger.delay_operator_statistics(
            self.short_name, is_succ, improvement_pct, elapsed_time_ms, new_weight)

        return g_new

    def apply(self, g: Graph, calculated_weights):
        raise NotImplementedError("This method has to be implemented in the class inheriting from Operator.")
