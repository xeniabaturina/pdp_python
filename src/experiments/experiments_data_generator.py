import random
from enum import Enum, auto

import numpy as np


def get_gaussian_districts_data(size_of_district=100):
    means = [
        (1, 1), (5, 3), (8, -3)
    ]
    covs = [
        [[1, 0], [0, 3]],
        [[1, 0], [0, 1]],
        [[5, 0], [0, 5]]
    ]

    lx, ly = [], []
    for i in range(len(means)):
        mean = means[i]
        cov = covs[i]

        x, y = np.random.multivariate_normal(mean, cov, (size_of_district,)).T

        lx += list(x)
        ly += list(y)

    x = np.array(lx)
    y = np.array(ly)

    x = x + abs(x.min())
    y = y + abs(y.min())

    x = x / x.max()
    y = y / y.max()

    return x, y


class OrdersModelType(Enum):
    MANY_TO_ONE = auto()
    ONE_TO_MANY = auto()
    NO_K_POINTS = auto()


def generate_coordinates(gaussian, points_num):
    coordinates = []

    if gaussian:
        x, y = get_gaussian_districts_data(size_of_district=(points_num // 3))

        for i, x_i in enumerate(x):
            coordinates.append((x_i, y[i]))

    print("SIZE " + str(len(coordinates)))

    return coordinates


def generate_orders(orders_model_type, points_num, orders_num, key_points_num=None):
    orders = []

    if key_points_num:
        key_points = [random.randrange(0, points_num) for _ in range(key_points_num)]

    for order in range(orders_num):
        if orders_model_type == OrdersModelType.MANY_TO_ONE:
            id_from = random.randrange(0, points_num)
            id_to = np.random.choice(key_points, size=1)[0]
        elif orders_model_type == OrdersModelType.ONE_TO_MANY:
            id_from = np.random.choice(key_points, size=1)[0]
            id_to = random.randrange(0, points_num)
        elif orders_model_type == OrdersModelType.NO_K_POINTS:
            id_from = random.randrange(0, points_num)
            id_to = random.randrange(0, points_num)
        orders.append((id_from, id_to))

    return orders
