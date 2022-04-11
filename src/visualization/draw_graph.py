from operator import itemgetter

import matplotlib
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Text

from ..graph_utils.Graph import Graph


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


def draw_path(edges, ax, g=None, gradient=False, normalize=True):
    params = {'facecolors': 'None', 'edgecolors': 'r'}

    count_of_edges = len(edges)
    if count_of_edges > 100:
        params['alpha'] = 0.3

    start_point = g.get_coords_by_id(g.edges[0][0]) if g and g.dataset_dir else edges[0][0]
    end_point = g.get_coords_by_id(g.edges[-1][1]) if g and g.dataset_dir else edges[-1][1]

    if normalize:
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

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

        color = cmap(cnt / len(edges) + 0.2) if gradient else 'g'
        if count_of_edges > 100:
            draw_arrow(ax, i, f, color=color, alpha=0.5)
        else:
            draw_arrow(ax, i, f, color=color)
        cnt += 1


def get_active_indexes(edges):
    indexes = []
    for edge in edges:
        indexes += edge
    indexes = list(dict.fromkeys(indexes))
    indexes.sort()
    return indexes


def normalize_coords(edges, coordinates_df):
    indexes = get_active_indexes(edges)

    x_min = np.array(itemgetter(*indexes)(coordinates_df.coord_x)).min()
    y_min = np.array(itemgetter(*indexes)(coordinates_df.coord_y)).min()
    x_min = abs(x_min) if x_min < 0 else -x_min
    y_min = abs(y_min) if y_min < 0 else -y_min

    x_new = coordinates_df.coord_x + x_min
    y_new = coordinates_df.coord_y + y_min

    x_new_max = np.array(itemgetter(*indexes)(x_new)).max()
    y_new_max = np.array(itemgetter(*indexes)(y_new)).max()

    factor = max(x_new_max, y_new_max)
    x_new = x_new / factor
    y_new = y_new / factor

    x_new_max = np.array(itemgetter(*indexes)(x_new)).max()
    y_new_max = np.array(itemgetter(*indexes)(y_new)).max()

    x_new = x_new + (1 - x_new_max) / 2
    y_new = y_new + (1 - y_new_max) / 2

    return pd.DataFrame({'coord_x': x_new, 'coord_y': y_new})


def draw_orders(ax, orders: List = None, dataset_dir: Text = None, normalize=True):
    if dataset_dir:
        orders_df = pd.read_csv(dataset_dir + '/orders.csv')
        coordinates_df = pd.read_csv(dataset_dir + '/coordinates_by_id.csv')
        orders_ids = orders_df[['from', 'to']].values.tolist()

        if normalize:
            coordinates_df = normalize_coords(orders_ids, coordinates_df)

        def get_coords(order):
            id_from, id_to = order
            x_coord_from = coordinates_df['coord_x'].values[id_from]
            y_coord_from = coordinates_df['coord_y'].values[id_from]
            x_coord_to = coordinates_df['coord_x'].values[id_to]
            y_coord_to = coordinates_df['coord_y'].values[id_to]
            return (x_coord_from, y_coord_from), (x_coord_to, y_coord_to)

        orders = list(map(lambda order: get_coords(order), orders_ids))
        draw_path(orders, ax)
    else:
        draw_path(orders, ax)


def draw_graph(ax, g: Graph, normalize=True):
    if normalize:
        g.coordinates_from_csv = normalize_coords(g.edges, g.coordinates_from_csv)

    draw_path(g.edges, ax, g, gradient=True)
