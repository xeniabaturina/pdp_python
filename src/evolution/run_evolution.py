import datetime
import imageio
import numpy as np
import os
import warnings

from matplotlib import pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from typing import AnyStr, Callable

from ..graph_utils.Graph import Graph

from ..multi_armed_bandit.Arms.CoupleExchange import CoupleExchange
from ..multi_armed_bandit.Arms.DoubleBridge import DoubleBridge
from ..multi_armed_bandit.Policies.EpsilonGreedy import EpsilonGreedy
from ..multi_armed_bandit.Arms.Lin2Opt import Lin2Opt
from ..multi_armed_bandit.MAB import MAB
from ..multi_armed_bandit.Arms.PointExchange import PointExchange
from ..multi_armed_bandit.Arms.RelocateBlock import RelocateBlock
from ..multi_armed_bandit.Result import Result

from ..evolution.evolutionary_algorithm import Evolution, fitness, TSP
from ..evolution.mutations import mab, to_string

from ..visualization.draw_graph import draw_graph

warnings.filterwarnings(action="ignore", module="scipy", message="^invalid value")


class EvolutionVisualisation:
    def __init__(self, evolution: Evolution, fitness_hist, time_hist, progress, time, epochs, pulls):
        self.evolution = evolution
        self.fitness_hist = fitness_hist
        self.time_hist = time_hist
        self.progress = progress
        self.time = time
        self.epochs = epochs
        self.pulls = pulls

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
        self.paths = {}


class Bandit:
    def __init__(self, policy, env, results):
        self.policy = policy
        self.env = env
        self.results = results
        self.time = 0

    def time_inc(self):
        self.time += 1


def run_evolution(
        n_epochs: int, pool_size: int, mutation: Callable,
        initial_graph: Graph,
        gif_filename: AnyStr = None):

    if gif_filename:
        gif_filename = gif_filename.replace('.gif', '')

    distances = Distances()
    bandit = None

    if mutation == mab:
        arm_configuration = [
            Lin2Opt(),
            DoubleBridge(),
            PointExchange(),
            CoupleExchange(),
            RelocateBlock(),
        ]

        policy = EpsilonGreedy(nb_arms=5, epsilon=0.3)
        env = MAB(arm_configuration)
        horizon = n_epochs * pool_size * 5
        results = Result(env.nbArms, horizon)
        bandit = Bandit(policy, env, results)

    evolution = Evolution(
        pool_size=pool_size,
        fitness=fitness,
        individual_class=TSP,
        mutation=mutation,
        initial_graph=initial_graph,
        distances=distances,
        bandit=bandit
    )

    fitness_hist = []
    time_hist = []
    filenames = []

    last_slope = np.Inf
    for epoch in (pbar := tqdm(range(n_epochs), leave=False)):
        best_result = evolution.get_best_weight()
        fitness_hist.append(-best_result)
        pbar.set_description(f'({to_string(mutation)})' +
                             f'Processing {epoch} epoch, {round(best_result, 2)}, slope {round(last_slope, 2)}')
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

        t = 5
        b = fitness_hist[-t:]
        a = [i for i, _ in enumerate(b)]

        eps = 1e-4
        last_slope = linregress(a, b).slope
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

    pulls = bandit.results.pulls if bandit else np.zeros(5)

    return EvolutionVisualisation(evolution, fitness_hist, time_hist, progress, time, epochs, pulls)
