import numpy as np
import pandas as pd

from enum import Enum
from os import path
from tqdm import tqdm
from typing import List, Tuple, AnyStr


class Mode(Enum):
    COORDS = 1
    IDS = 2
    EDGES = 3


class Graph:
    def __init__(
            self,
            mode: Mode = Mode.COORDS,
            data: List[Tuple] = None,
            dataset_dir: AnyStr = None):

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.weight = None
        self.coordinates_from_csv = None

        if mode == Mode.COORDS:
            if data is None:
                raise ValueError('need coordinates to process graph')
            self.edges = self.construct_by_coords(data)

        elif mode == Mode.IDS:
            if dataset_dir is None:
                raise ValueError('need dataset to process graph on it')

            if not(path.exists(dataset_dir + '/orders.csv')):
                raise FileExistsError('directory ' + dataset_dir + ' does not exist')
            self.edges = self.construct_by_dataset(dataset_dir)

        elif mode == Mode.EDGES:
            if data is None:
                raise ValueError('need edges to process graph')

            self.edges = data

        else:
            self.edges = []

    @staticmethod
    def construct_by_coords(points):
        p = points
        np.random.shuffle(p)
        edges = []

        for i, _ in enumerate(p[:-1]):
            edges.append(
                (p[i], p[i + 1])
            )

        return edges

    @staticmethod
    def construct_by_dataset(dataset_dir):
        edges = []
        orders_df = pd.read_csv(dataset_dir + '/orders.csv')
        p = [*orders_df['from'].values.tolist(), *orders_df['to'].values.tolist()]

        for i, _ in tqdm(enumerate(p[:-1])):
            edges.append(
                (p[i], p[i + 1])
            )

        return edges

    def get_coords_by_id(self, idx):
        if self.coordinates_from_csv is None:
            dataset_dir = self.dataset_dir

            if dataset_dir is None:
                raise ValueError('graph has no dataset')

            self.coordinates_from_csv = pd.read_csv(dataset_dir + '/coordinates_by_id.csv', sep=',')

        return np.array(self.coordinates_from_csv.iloc[idx].to_list())

    def get_edge_weight(self, edge, distances=None):
        v1, v2 = edge

        if self.dataset_dir:
            if distances:
                if (v1, v2) in distances.distances:
                    return distances.distances[(v1, v2)]

                else:
                    x1, y1 = self.get_coords_by_id(v1)
                    x2, y2 = self.get_coords_by_id(v2)
                    weight = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    distances.distances[(v1, v2)] = weight
                    distances.distances[(v2, v1)] = weight

                    return weight

            else:
                x1, y1 = self.get_coords_by_id(v1)
                x2, y2 = self.get_coords_by_id(v2)

                return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        else:
            return np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

    def compute_path_weight(self, distances=None):
        weight = sum([
            self.get_edge_weight(edge, distances)
            for edge in self.edges
        ])

        self.weight = weight
        return weight

    def get_path_weight(self, distances=None):
        if self.weight is not None:
            return self.weight
        else:
            return self.compute_path_weight(distances)

    def set_path_weight(self, g_new_weight):
        self.weight = g_new_weight

    def is_valid(self):
        dataset_dir = self.dataset_dir

        if dataset_dir:
            orders_df = pd.read_csv(dataset_dir + '/orders.csv', sep=',')
            orders = orders_df[['from', 'to']].values.tolist()
            current_path_list = [*list(edge[0] for edge in self.edges), self.edges[-1][1]]
            current_path = ','.join(str(x) for x in current_path_list)
            for order in orders:
                id_from, id_to = order
                if current_path.index(str(id_from)) > current_path.rindex(str(id_to)):
                    return False
            return True
        else:
            return True

    def optimize(self):
        edges = list(filter(lambda v: v[0] != v[1], self.edges))
        optimized_g = Graph(Mode.EDGES, edges, self.dataset_dir)
        optimized_g.coordinates_from_csv = self.coordinates_from_csv
        return optimized_g
