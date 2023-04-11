import datetime
import imageio
import numpy as np
import os
import warnings

from matplotlib import pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from typing import AnyStr

from ..graph_utils.Graph import Graph

from ..evolution.evolutionary_algorithm import Evolution, fitness, TSP

from ..visualization.draw_graph import draw_graph

warnings.filterwarnings(action="ignore", module="scipy", message="^invalid value")


class EvolutionVisualisation:
    def __init__(self, evolution: Evolution, fitness_hist, time_hist, progress, time, epochs, pulls, pulls_succ,
                 probabilities):
        self.probabilities = probabilities
        self.evolution = evolution
        self.fitness_hist = fitness_hist
        self.time_hist = time_hist
        self.progress = progress
        self.time = time
        self.epochs = epochs
        self.pulls = pulls
        self.pulls_succ = pulls_succ

    def get_probabilities(self):
        return self.probabilities

    def get_fitness_hist(self):
        return self.fitness_hist

    def get_time_hist(self):
        return self.time_hist

    def get_best_path(self):
        return self.evolution.get_best_path()

    def get_best_weight(self):
        return self.evolution.get_best_weight()


class Distances:
    def __init__(self):
        self.distances = {}


def run_evolution(
        logger,
        n_epochs: int, pool_size: int,
        offsprings_num,
        initial_graph: Graph,
        strategy,
        evolution_type,
        derivative_bound=0,
        gif_filename: AnyStr = None, log="",
        tqdm_disable=False):

    if gif_filename:
        gif_filename = gif_filename.replace('.gif', '')

    calculated_weights = Distances()

    evolution = Evolution(
        pool_size=pool_size,
        offsprings_num=offsprings_num,
        fitness=fitness,
        evolution_type=evolution_type,
        individual_class=TSP,
        initial_graph=initial_graph,
        calculated_weights=calculated_weights,
        strategy=strategy,
        logger=logger
    )

    fitness_hist = []
    probabilities = []
    time_hist = []
    filenames = []

    last_slope = np.Inf
    for epoch in (pbar := tqdm(range(n_epochs), leave=False, disable=tqdm_disable)):
        best_result = evolution.get_best_weight()
        fitness_hist.append(-best_result)

        probabilities.append(strategy.probabilities)

        pbar.set_description(f'{log}' + f' | Processing {epoch} epoch, {round(best_result, 2)}, slope {round(last_slope, 2)}')
        evolution.step()

        if gif_filename:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_facecolor('#fcfcfc')
            draw_graph(ax, evolution.get_best_g())

            # create file name and append it to a list
            filename = f'src/visualization/tmp/{epoch}.png'
            filenames.append(filename)

            # save frame
            plt.savefig(filename)
            plt.close()

        time_hist.append(pbar.format_dict["elapsed"])

        t = 25
        b = fitness_hist[-t:]
        a = [i for i, _ in enumerate(b)]

        eps = derivative_bound
        last_slope = linregress(a, b).slope

        if derivative_bound > 0:
            if epoch >= t and last_slope < eps:
                break

    start_path = -round(fitness_hist[0], 2)
    end_path = -round(fitness_hist[-1], 2)

    progress = round((1 - end_path / start_path) * 100, 2)
    time = datetime.timedelta(seconds=round(time_hist[-1]))
    epochs = len(time_hist)

    if gif_filename:
        # build gif
        with imageio.get_writer('figures/gifs/' + gif_filename + '.gif', mode='I', fps=6) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in set(filenames):
            os.remove(filename)

    pulls = strategy.pulls
    pulls_succ = strategy.pulls_succ

    return EvolutionVisualisation(evolution, fitness_hist, time_hist, progress, time, epochs, pulls, pulls_succ, probabilities)
