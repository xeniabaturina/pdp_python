import numpy as np
import pandas as pd

from src.graph_utils.Graph import Graph
from src.graph_utils.Graph import Mode
from src.logs_utils.Logger import Logger
from src.operators_utils.Operator import Operator


class Lin2Opt(Operator):
    def __init__(self, logger: Logger):
        super(Lin2Opt, self).__init__('lin2opt', logger)

    def apply(self, g: Graph, calculated_weights):
        edges = g.edges[:]
        g_old_weight = g.get_path_weight(calculated_weights)
        g_new_weight = g_old_weight

        k = np.random.randint(0, len(edges) - 2)
        j = np.random.randint(k + 2, len(edges))

        g_new_weight -= (
                g.get_edge_weight(edges[k], calculated_weights)
                + g.get_edge_weight(edges[j], calculated_weights)
        )

        k_v1, k_v2 = edges[k]
        j_v1, j_v2 = edges[j]

        edges[k] = (k_v1, j_v1)
        edges[j] = (k_v2, j_v2)

        for i in range(k + 1, j):
            i_v1, i_v2 = edges[i]
            edges[i] = (i_v2, i_v1)

        reversed_edges = edges[k + 1: j]
        reversed_edges.reverse()

        for i in range(k + 1, j):
            edges[i] = reversed_edges[i - k - 1]

        g_new_weight += (
                g.get_edge_weight(edges[k], calculated_weights)
                + g.get_edge_weight(edges[j], calculated_weights)
        )

        g_new = Graph(Mode.EDGES, edges, g.dataset_dir)
        g_new.set_path_weight(g_new_weight)
        g_new.coordinates_from_csv = g.coordinates_from_csv

        return g_new


class DoubleBridge(Operator):
    def __init__(self, logger: Logger):
        super(DoubleBridge, self).__init__('double_bridge', logger)

    def apply(self, g: Graph, calculated_weights):
        edges = g.edges[:]
        g_old_weight = g.get_path_weight(calculated_weights)
        g_new_weight = g_old_weight

        j = np.random.randint(0, len(edges) - 6)
        k = np.random.randint(j + 2, len(edges) - 4)
        l = np.random.randint(k + 2, len(edges) - 2)
        h = np.random.randint(l + 2, len(edges))

        g_new_weight -= (
                g.get_edge_weight(edges[j], calculated_weights)
                + g.get_edge_weight(edges[k], calculated_weights)
                + g.get_edge_weight(edges[l], calculated_weights)
                + g.get_edge_weight(edges[h], calculated_weights)
        )

        j_v1, j_v2 = edges[j]
        k_v1, k_v2 = edges[k]
        l_v1, l_v2 = edges[l]
        h_v1, h_v2 = edges[h]

        edges[j] = (j_v1, l_v2)
        edges[k] = (k_v1, h_v2)
        edges[l] = (l_v1, j_v2)
        edges[h] = (h_v1, k_v2)

        g_new_weight += (
                g.get_edge_weight(edges[j], calculated_weights)
                + g.get_edge_weight(edges[k], calculated_weights)
                + g.get_edge_weight(edges[l], calculated_weights)
                + g.get_edge_weight(edges[h], calculated_weights)
        )

        start_to_j = edges[:j]
        j_to_k = edges[j + 1:k]
        k_to_l = edges[k + 1:l]
        l_to_h = edges[l + 1:h]
        h_to_end = edges[h + 1:]

        g_new = Graph(Mode.EDGES,
                      [*start_to_j, edges[j],
                       *l_to_h, edges[h],
                       *k_to_l, edges[l],
                       *j_to_k, edges[k],
                       *h_to_end],
                      g.dataset_dir)
        g_new.set_path_weight(g_new_weight)
        g_new.coordinates_from_csv = g.coordinates_from_csv

        return g_new


class PointExchange(Operator):
    def __init__(self, logger: Logger):
        super(PointExchange, self).__init__('point_exchange', logger)

    def apply(self, g: Graph, calculated_weights):
        edges = g.edges[:]
        g_old_weight = g.get_path_weight(calculated_weights)
        g_new_weight = g_old_weight

        j = np.random.randint(0, len(edges) - 2)
        k = np.random.randint(j + 1, len(edges) - 1)
        h = k + 1

        g_new_weight -= \
            g.get_edge_weight(edges[j], calculated_weights) + \
            g.get_edge_weight(edges[k], calculated_weights) + \
            g.get_edge_weight(edges[h], calculated_weights)

        if k == j + 1:
            j_v1, j_v2 = edges[j]
            k_v1, k_v2 = edges[k]
            h_v1, h_v2 = edges[h]

            edges[j] = (j_v1, h_v1)
            edges[k] = (h_v1, k_v1)
            edges[h] = (k_v1, h_v2)
        else:
            l = j + 1
            g_new_weight -= g.get_edge_weight(edges[l], calculated_weights)

            j_v1, j_v2 = edges[j]
            l_v1, l_v2 = edges[l]
            k_v1, k_v2 = edges[k]
            h_v1, h_v2 = edges[h]

            edges[j] = (j_v1, h_v1)
            edges[l] = (h_v1, l_v2)
            edges[k] = (k_v1, l_v1)
            edges[h] = (l_v1, h_v2)

            g_new_weight += g.get_edge_weight(edges[l], calculated_weights)

        g_new_weight += (
                g.get_edge_weight(edges[j], calculated_weights)
                + g.get_edge_weight(edges[k], calculated_weights)
                + g.get_edge_weight(edges[h], calculated_weights)
        )

        g_new = Graph(Mode.EDGES, edges, g.dataset_dir)
        g_new.set_path_weight(g_new_weight)
        g_new.coordinates_from_csv = g.coordinates_from_csv

        return g_new


class CoupleExchange(Operator):
    def __init__(self, logger: Logger):
        super(CoupleExchange, self).__init__('couple_exchange', logger)

    def apply(self, g: Graph, calculated_weights):
        edges = g.edges[:]
        dataset_dir = g.dataset_dir
        g_old_weight = g.get_path_weight(calculated_weights)
        g_new_weight = g_old_weight

        if dataset_dir:
            orders_df = pd.read_csv(dataset_dir + '/orders.csv')
            current_path = [*list(edge[0] for edge in edges), edges[-1][1]]
            orders = orders_df[['from', 'to']].values.tolist()

            k = np.random.randint(0, len(orders) - 1)
            j = np.random.randint(k, len(orders))

            condition = current_path.index(orders[k][1]) < current_path.index(orders[j][0])

            (from1, to1), (from2, to2) = (orders[k], orders[j]) if condition else (orders[j], orders[k])

            if current_path.index(to1) <= current_path.index(from2):
                from1_index = current_path.index(from1)
                to1_index = current_path.index(to1)
                from2_index = current_path.index(from2)
                to2_index = current_path.index(to2)

                start_to_first = edges[:from1_index]
                first = edges[from1_index:to1_index]
                first_to_second = edges[to1_index:from2_index]
                second = edges[from2_index:to2_index]
                second_to_end = edges[to2_index:]

                if len(first) > 0 and len(second) > 0 and len(first_to_second) > 0:
                    if len(start_to_first) > 0:
                        g_new_weight -= g.get_edge_weight(start_to_first[-1], calculated_weights)
                        start_to_first[-1] = start_to_first[-1][0], second[0][0]
                        g_new_weight += g.get_edge_weight(start_to_first[-1], calculated_weights)

                    if len(second_to_end) > 0 and len(first) > 0:
                        g_new_weight -= g.get_edge_weight(second_to_end[0], calculated_weights)
                        second_to_end[0] = first[-1][1], second_to_end[0][1]
                        g_new_weight += g.get_edge_weight(second_to_end[0], calculated_weights)

                    g_new_weight -= g.get_edge_weight(first_to_second[0], calculated_weights)
                    g_new_weight -= g.get_edge_weight(first_to_second[-1], calculated_weights)

                    first_to_second[0] = second[-1][1], first_to_second[0][1]
                    first_to_second[-1] = first_to_second[-1][0], first[0][0]

                    g_new_weight += g.get_edge_weight(first_to_second[0], calculated_weights)
                    g_new_weight += g.get_edge_weight(first_to_second[-1], calculated_weights)

                    edges = [*start_to_first,
                             *second,
                             *first_to_second,
                             *first,
                             *second_to_end]

        g_new = Graph(Mode.EDGES, edges, g.dataset_dir)
        g_new.set_path_weight(g_new_weight)
        g_new.coordinates_from_csv = g.coordinates_from_csv

        return g_new


class RelocateBlock(Operator):
    def __init__(self, logger: Logger):
        super(RelocateBlock, self).__init__('relocate_block', logger)

    def apply(self, g: Graph, calculated_weights):
        edges = g.edges[:]
        g_old_weight = g.get_path_weight(calculated_weights)

        j = np.random.randint(0, len(edges) - 2)
        k = np.random.randint(j + 1, len(edges) - 1)

        start_to_j = edges[:j]
        j_to_k = edges[j:k]
        k_to_end = edges[k:]

        first_from_tail = k_to_end[0][0]
        kj_old_weight = 0
        if len(start_to_j) > 0:
            kj_old_weight += g.get_edge_weight(start_to_j[-1], calculated_weights)
            start_to_j[-1] = start_to_j[-1][0], first_from_tail

        others = [*start_to_j, *k_to_end]

        best = edges
        g_best_weight = g_old_weight
        kj_old_weight += g.get_edge_weight(j_to_k[-1], calculated_weights)

        if len(start_to_j) > 0:
            kj_old_weight -= g.get_edge_weight(start_to_j[-1], calculated_weights)

        for i in range(len(others)):

            head = others[:i]
            tail = others[i:]

            kj_new_weight = 0
            if len(head) > 0:
                kj_new_weight -= g.get_edge_weight(head[-1], calculated_weights)
                head[-1] = head[-1][0], j_to_k[0][0]

            j_to_k[-1] = j_to_k[-1][0], tail[0][0]

            if len(head) > 0:
                kj_new_weight += g.get_edge_weight(head[-1], calculated_weights)

            kj_new_weight += g.get_edge_weight(j_to_k[-1], calculated_weights)
            g_tmp_weight = g_old_weight - kj_old_weight + kj_new_weight

            if g_tmp_weight < g_best_weight:
                g_best_weight = g_tmp_weight
                best = [*head, *j_to_k, *tail]

        g_new = Graph(Mode.EDGES, best, g.dataset_dir)
        g_new.set_path_weight(g_best_weight)
        g_new.coordinates_from_csv = g.coordinates_from_csv

        return g_new


class RandomRelocateBlock(Operator):
    def __init__(self, logger: Logger):
        super(RandomRelocateBlock, self).__init__('rnd_relocate_block', logger)

    def apply(self, g: Graph, calculated_weights):
        edges = g.edges[:]
        g_old_weight = g.get_path_weight(calculated_weights)

        j = np.random.randint(0, len(edges) - 2)
        k = np.random.randint(j + 1, len(edges) - 1)

        start_to_j = edges[:j]
        j_to_k = edges[j:k]
        k_to_end = edges[k:]

        first_from_tail = k_to_end[0][0]
        kj_old_weight = 0
        if len(start_to_j) > 0:
            kj_old_weight += g.get_edge_weight(start_to_j[-1], calculated_weights)
            start_to_j[-1] = start_to_j[-1][0], first_from_tail

        others = [*start_to_j, *k_to_end]

        best = edges
        g_best_weight = g_old_weight
        kj_old_weight += g.get_edge_weight(j_to_k[-1], calculated_weights)

        if len(start_to_j) > 0:
            kj_old_weight -= g.get_edge_weight(start_to_j[-1], calculated_weights)

        i = np.random.randint(0, len(others))

        head = others[:i]
        tail = others[i:]

        kj_new_weight = 0
        if len(head) > 0:
            kj_new_weight -= g.get_edge_weight(head[-1], calculated_weights)
            head[-1] = head[-1][0], j_to_k[0][0]

        j_to_k[-1] = j_to_k[-1][0], tail[0][0]

        if len(head) > 0:
            kj_new_weight += g.get_edge_weight(head[-1], calculated_weights)

        kj_new_weight += g.get_edge_weight(j_to_k[-1], calculated_weights)
        g_tmp_weight = g_old_weight - kj_old_weight + kj_new_weight

        if g_tmp_weight < g_best_weight:
            g_best_weight = g_tmp_weight
            best = [*head, *j_to_k, *tail]

        g_new = Graph(Mode.EDGES, best, g.dataset_dir)
        g_new.set_path_weight(g_best_weight)
        g_new.coordinates_from_csv = g.coordinates_from_csv

        return g_new


class Combined(Operator):
    def __init__(self, logger: Logger, op1: Operator, op2: Operator):
        super(Combined, self).__init__(op1.short_name + '-' + op2.short_name, logger)

        self.op1 = op1
        self.op2 = op2

    def apply(self, g: Graph, calculated_weights):
        g_tmp = self.op1.apply(g, calculated_weights)
        g_new = self.op2.apply(g_tmp, calculated_weights)
        return g_new
