import matplotlib
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Text

from src.graph_utils.Graph import Graph


def vec_norm(v):
    return np.sqrt((v[0]) ** 2 + (v[1]) ** 2)


def vector(p1, p2):
    return np.array([p2[0] - p1[0], (p2[1] - p1[1])])


def draw_arrow(ax, p1, p2, **kwargs):
    ax.arrow(
        p1[0], p1[1],
        p2[0] - p1[0], p2[1] - p1[1],
        transform=ax.transData,
        width=0.005,
        head_length=0.03,
        head_width=0.03,
        **kwargs
    )


def draw_path(edges, ax, g=None, gradient=False):
    params = {'facecolors': 'None', 'edgecolors': 'r'}

    start_point = g.get_coords_by_id(g.edges[0][0]) if g and g.dataset_dir else edges[0][0]
    end_point = g.get_coords_by_id(g.edges[-1][1]) if g and g.dataset_dir else edges[-1][1]

    ax.scatter(*start_point, s=100, **params)
    ax.scatter(*end_point, s=100, **params)

    cmap = matplotlib.cm.get_cmap('Blues')
    cnt = 1
    for v1, v2 in tqdm(edges):
        if g and g.dataset_dir:
            v1 = g.get_coords_by_id(v1)
            v2 = g.get_coords_by_id(v2)

        ax.scatter(*v1, s=100, **params)
        ax.scatter(*v2, s=100, **params)

        vec = vector(v1, v2)
        center = np.array(v1) + vec / 2
        i = center - (vec / 2) * 0.9
        f = center + (vec / 2) * 0.8

        color = cmap(cnt / len(edges) + 0.2)if gradient else 'g'
        draw_arrow(ax, i, f, color=color)
        cnt += 1


def draw_orders(ax, orders: List = None, dataset_dir: Text = None):
    if dataset_dir:
        orders_df = pd.read_csv('../../data/' + dataset_dir + '/orders.csv')
        coordinates_df = pd.read_csv('../../data/' + dataset_dir + '/coordinates_by_id.csv')
        orders_ids = orders_df[['from', 'to']].values.tolist()

        def get_coords(order):
            id_from, id_to = order
            x_coord_from = coordinates_df['coord_x'].values[id_from]
            y_coord_from = coordinates_df['coord_y'].values[id_from]
            x_coord_to = coordinates_df['coord_x'].values[id_to]
            y_coord_to = coordinates_df['coord_y'].values[id_to]
            return (x_coord_from, y_coord_from), (x_coord_to, y_coord_to)

        orders = list(map(lambda order: get_coords(order), orders_ids))

    draw_path(orders, ax)


def draw_graph(ax, g: Graph):
    draw_path(g.edges, ax, g, gradient=True)
