import numpy as np


def spiral_archimedes(k, r_start=0, r_end=1, spins=3, random=False):
    phi = np.linspace(0, 2 * np.pi * spins, k)
    r = np.linspace(r_start, r_end, k)

    x = (r * np.cos(phi) + 1) / 2
    y = (r * np.sin(phi) + 1) / 2

    if random:
        x, y = x + np.random.normal(size=k) / 50, y + np.random.normal(size=k) / 50

    return list(zip(x, y))


def random_points(size: int) -> list:
    return np.random.rand(size * 2).reshape(2, size).T
