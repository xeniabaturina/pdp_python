import ast
from enum import Enum
from pathlib import Path

import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

all_evol_algorithms = ('1+1', '1,1', '1+N', '1,N', 'K+KN', 'K,KN')
all_operators = ('lin2opt', 'double_bridge', 'point_exchange', 'couple_exchange', 'relocate_block',
                 'rnd_relocate_block',
                 'lin2opt-lin2opt', 'lin2opt-double_bridge', 'lin2opt-point_exchange',
                 'lin2opt-couple_exchange', 'lin2opt-relocate_block', 'lin2opt-rnd_relocate_block',
                 'double_bridge-lin2opt', 'double_bridge-double_bridge', 'double_bridge-point_exchange',
                 'double_bridge-couple_exchange', 'double_bridge-relocate_block', 'double_bridge-rnd_relocate_block',
                 'point_exchange-lin2opt', 'point_exchange-double_bridge', 'point_exchange-point_exchange',
                 'point_exchange-couple_exchange', 'point_exchange-relocate_block', 'point_exchange-rnd_relocate_block',
                 'couple_exchange-lin2opt', 'couple_exchange-double_bridge', 'couple_exchange-point_exchange',
                 'couple_exchange-couple_exchange', 'couple_exchange-relocate_block',
                 'couple_exchange-rnd_relocate_block',
                 'relocate_block-lin2opt', 'relocate_block-double_bridge', 'relocate_block-point_exchange',
                 'relocate_block-couple_exchange', 'relocate_block-relocate_block', 'relocate_block-rnd_relocate_block',
                 'rnd_relocate_block-lin2opt', 'rnd_relocate_block-double_bridge', 'rnd_relocate_block-point_exchange',
                 'rnd_relocate_block-couple_exchange', 'rnd_relocate_block-relocate_block',
                 'rnd_relocate_block-rnd_relocate_block')


def get_list_of_all_experiments():
    experiments_logs_path = Path('../../data/experiments_logs')
    experiments_paths = [str(x) for x in experiments_logs_path.glob('*') if x.is_dir()]
    return experiments_paths


def get_sub_experiments_by_name(experiment_name):
    p = re.compile('^' + experiment_name + r'_\d*$')

    experiments = [s for s in list(map(os.path.basename, get_list_of_all_experiments())) if p.match(s)]

    if len(experiments) == 0:
        p = re.compile(r'^' + experiment_name + '$')
        experiments = [s for s in list(map(os.path.basename, get_list_of_all_experiments())) if p.match(s)]

    return experiments


def get_bounds_of_sub_experiment(sub_experiment_name, sub_experiment_path):
    p = re.compile('^' + sub_experiment_name + r'_\d*_\d*_(.*?)_[a-z]*$')

    bounds_paths = [x for x in sub_experiment_path.glob('*') if x.is_dir()]

    bounds_with_dir_names = [(p.match(s).group(1), p.match(s).group(0)) for s in
                             list(map(os.path.basename, bounds_paths)) if
                             p.match(s)]

    return bounds_with_dir_names


def get_evol_algorithms_of_sub_experiment(sub_experiment_path, bound_dir_name):
    bound_dir_path = sub_experiment_path / bound_dir_name
    evol_algorithms = [x.stem for x in bound_dir_path.glob('*') if x.is_dir()]

    evol_algorithms_sorted = [x for x in all_evol_algorithms if x in evol_algorithms]

    return evol_algorithms_sorted


def get_strategies_of_sub_experiment(simulation_details_path):
    p = re.compile('^s_')

    strategies = [x.stem for x in simulation_details_path.glob('*') if x.is_file()]

    strategies.sort(key=lambda x: x.startswith("s_"), reverse=False)

    return strategies


def get_simulation_details_paths(sub_experiment_path, bound_dir_name, evol_algorithm):
    details_path = sub_experiment_path / bound_dir_name / evol_algorithm / 'details'

    return [x for x in details_path.glob('*') if x.is_dir()]


def get_simulation_operators(sub_experiment_operators_logs_path, bound_dir_name, evol_algorithm):
    operators_path = sub_experiment_operators_logs_path / bound_dir_name / evol_algorithm

    return [x.stem for x in operators_path.glob('*') if x.is_file()]


def get_experiment_parameters(experiment_name):
    sub_experiments_names = get_sub_experiments_by_name(experiment_name)
    first_sub_experiment_name = sub_experiments_names[0]

    experiments_logs_path = Path('../../data/experiments_logs/')
    operators_logs_path = Path('../../data/operators_logs/')

    first_sub_experiment_path = experiments_logs_path / first_sub_experiment_name

    bounds_with_dir_names = get_bounds_of_sub_experiment(first_sub_experiment_name, first_sub_experiment_path)

    first_bound_dir_name = bounds_with_dir_names[0][1]
    # first_evol_algorithm = evol_algorithms[0]
    evol_algorithms = get_evol_algorithms_of_sub_experiment(first_sub_experiment_path, first_bound_dir_name)
    first_evol_algorithm = evol_algorithms[0]

    simulation_details_paths = get_simulation_details_paths(first_sub_experiment_path, first_bound_dir_name,
                                                            first_evol_algorithm)

    strategies = get_strategies_of_sub_experiment(simulation_details_paths[0])

    first_sub_experiment_operators_logs_path = operators_logs_path / first_sub_experiment_name

    operators = get_simulation_operators(first_sub_experiment_operators_logs_path,
                                         first_bound_dir_name, first_evol_algorithm)

    return strategies, evol_algorithms, operators


class FeaturesType(Enum):
    FITNESS = 'fitness'
    PROBABILITIES = 'probabilities'


def get_fitness_from_str(s, _):
    return [float(x) for x in s.split('#')]


def get_probs_from_str(s, operators):
    probs_by_mutation = {mutation: [] for mutation in operators}
    probs_str_lst = list(filter(None, s.split('&')))

    for probs_str in probs_str_lst:
        probs_lst = []
        for x in probs_str.split('#'):
            probs_lst.append(ast.literal_eval(x.replace('nan', '0')))

        # probs_lst = [ast.literal_eval(x) for x in probs.split('#')]

        for probs in probs_lst:
            for idx, prob in enumerate(probs):
                probs_by_mutation[operators[idx]].append(float(prob))

    return probs_by_mutation


def get_features_by_sub_experiment(sub_experiment_name, sub_experiment_path, features_type, strategies, evol_algorithms,
                                   operators):
    bounds_with_dir_names = get_bounds_of_sub_experiment(sub_experiment_name, sub_experiment_path)

    features_hists_by_bound = {}

    for (bound, _) in bounds_with_dir_names:
        features_hists_by_bound[bound] = {}

    for evol_algorithm in evol_algorithms:
        for (bound, bound_dir_name) in bounds_with_dir_names:

            details_path = sub_experiment_path / bound_dir_name / evol_algorithm / 'details'

            simulations_paths = [x for x in details_path.glob('*') if x.is_dir()]

            strategies_features = {strategy: [] for strategy in strategies}
            for inner_experiment_path in simulations_paths:
                for strategy in strategies:
                    strategy_df = pd.read_csv(inner_experiment_path / (strategy + '.csv'), sep=',')

                    features_from_str = (get_fitness_from_str if features_type == FeaturesType.FITNESS
                                         else get_probs_from_str)
                    column_name = 'fitness_hist' if features_type == FeaturesType.FITNESS else 'probabilities'

                    strategies_features[strategy] += [features_from_str(s, operators)
                                                      for s in strategy_df[column_name].tolist()]

            features_hists_by_bound[bound][evol_algorithm] = strategies_features

    return features_hists_by_bound


def get_features_by_experiment(experiment_name, features_type, strategies, evol_algorithms, operators):
    f_hists_by_experiment = []
    sub_experiments_names = get_sub_experiments_by_name(experiment_name)

    experiments_logs_path = Path('../../data/experiments_logs/')

    for sub_experiment_name in sub_experiments_names:
        sub_experiment_path = experiments_logs_path / sub_experiment_name

        f_hists_by_sub_experiment = get_features_by_sub_experiment(sub_experiment_name, sub_experiment_path,
                                                                   features_type, strategies, evol_algorithms,
                                                                   operators)
        f_hists_by_experiment.append(f_hists_by_sub_experiment)

    return f_hists_by_experiment


def average(lst):
    return sum(lst) / len(lst)


def get_mean_sub_hist(global_mean_hist, f_hists, features_type, mutation=None):
    max_len = max(len(
        hist if features_type == FeaturesType.FITNESS else hist[mutation]
    ) for hist in f_hists)

    for i in range(max_len):
        global_mean_hist.append(average([(hist if features_type == FeaturesType.FITNESS else hist[mutation])[i]
                                         for hist in f_hists
                                         if len(hist
                                                if features_type == FeaturesType.FITNESS
                                                else hist[mutation]) > i]))


def get_mean_hist(f_hists, features_type, operators):
    mean_hist = [] if features_type == FeaturesType.FITNESS else {mutation: [] for mutation in operators}

    if features_type == FeaturesType.FITNESS:
        get_mean_sub_hist(mean_hist, f_hists, features_type)
    else:
        for mutation in operators:
            get_mean_sub_hist(mean_hist[mutation], f_hists, features_type, mutation)

    return mean_hist


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def plot_features_hists_by_bandit(features_by_bandit, evol_algorithm, features_type, bound, ax, strategies, operators,
                                  mean=False, alpha=0.5, bandit=None, epochs_limit=None):
    colors = [plt.cm.tab10(i) for i in range(max(len(strategies), len(operators)))]

    ax.set_title(evol_algorithm, fontsize=20, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=20)

    if mean:
        ax.set_ylabel('mean ' + features_type.value, fontsize=20)
    else:
        ax.set_ylabel(features_type.value, fontsize=20)

    features = strategies if features_type == FeaturesType.FITNESS else operators

    for idx, bandit_or_mutation in enumerate(features):
        current_bandit = bandit_or_mutation if features_type == FeaturesType.FITNESS else bandit
        if mean:
            f_hists = [features_by_bandit[idx_sub_experiment][bound][evol_algorithm][current_bandit]
                       for idx_sub_experiment in range(len(features_by_bandit))]
            mean_hist = get_mean_hist(flatten(f_hists), features_type, operators)
            ax.plot((mean_hist if features_type == FeaturesType.FITNESS
                     else mean_hist[bandit_or_mutation]), color=colors[idx], alpha=alpha)
        else:
            for idx_sub_experiment in range(len(features_by_bandit)):
                for hist in features_by_bandit[idx_sub_experiment][bound][evol_algorithm][current_bandit]:
                    ax.plot((hist if features_type == FeaturesType.FITNESS
                             else hist[bandit_or_mutation]), color=colors[idx], alpha=alpha)


def get_limits_by_bandit(f_hists_by_bandit, bound, evol_algorithm, bandit, operators, features_type, xmax, ymin, ymax,
                         epochs_limit=None):
    for idx_sub_experiment in range(len(f_hists_by_bandit)):
        for hist in f_hists_by_bandit[idx_sub_experiment][bound][evol_algorithm][bandit]:
            if features_type == FeaturesType.FITNESS:
                if not epochs_limit:
                    xmax = max(xmax, len(hist))
                ymin = min(ymin, min(hist))
                ymax = max(ymax, max(hist))
            else:
                for mutation in operators:
                    if not epochs_limit:
                        xmax = max(xmax, len(hist[mutation]))

    return xmax, ymin, ymax


def get_limits(f_hists_by_bandit, features_type, bound, strategies, evol_algorithms, operators, bandit=None,
               epochs_limit=None):
    xmin, xmax = (0, epochs_limit) if epochs_limit else (0, 0)
    ymin, ymax = ((np.inf, -np.inf) if features_type == FeaturesType.FITNESS else (0, 1))

    # for bound in bounds:
    for evol_algorithm in evol_algorithms:
        if features_type == FeaturesType.FITNESS:
            for idx, bandit in enumerate(strategies):
                xmax, ymin, ymax = get_limits_by_bandit(f_hists_by_bandit, bound, evol_algorithm, bandit, operators,
                                                        features_type, xmax, ymin, ymax, epochs_limit)
        else:
            xmax, ymin, ymax = get_limits_by_bandit(f_hists_by_bandit, bound, evol_algorithm, bandit, operators,
                                                    features_type, xmax, ymin, ymax, epochs_limit)

    return xmin, xmax, ymin, ymax


def plot_features_by_experiment_and_bound(features_type, experiment_name, bound, mean=False, alpha=0.5, bandit=None,
                                          horizontal=False, epochs_limit=None):
    strategies, evol_algorithms, operators = get_experiment_parameters(experiment_name)
    features_by_experiment = get_features_by_experiment(experiment_name, features_type, strategies, evol_algorithms,
                                                        operators)

    plt.rcParams["font.family"] = "Times New Roman"

    nrows, ncols = (6, 1) if horizontal else (3, 2)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12/1.3, 18/1.3))
    # fig.tight_layout(h_pad=5, w_pad=5)

    # top=0.99, bottom=0.01
    plt.subplots_adjust(hspace=0.45, wspace=0.3)

    title = experiment_name + ", bound: " + bound
    if bandit:
        title += ", bandit: " + bandit
    fig.suptitle(title, fontsize=20, fontweight='bold')

    xmin, xmax, ymin, ymax = get_limits(features_by_experiment, features_type, bound, strategies, evol_algorithms,
                                        operators, bandit, epochs_limit)

    for idx, evol_algorithm in enumerate(evol_algorithms):
        ax_xy = idx if horizontal else (idx // ncols, idx % ncols)
        ax[ax_xy].set_xlim([xmin, xmax])
        ax[ax_xy].set_ylim([ymin, ymax])
        plot_features_hists_by_bandit(features_by_experiment,
                                      evol_algorithm, features_type, bound,
                                      ax[ax_xy], strategies, operators, mean, alpha, bandit, epochs_limit)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax[ax_xy].spines[axis].set_linewidth(1.5)
        ax[ax_xy].tick_params(direction="in", width=1.5, length=4)

    features = strategies if features_type == FeaturesType.FITNESS else operators

    colors = [plt.cm.tab10(i) for i in range(max(len(strategies), len(operators)))]
    custom_lines = [Line2D([0], [0], color=colors[idx], lw=2, alpha=0.5)
                    for idx, _ in enumerate(features)]

    fig.legend(custom_lines, features, fontsize=12)

    plots_path = Path('../../data/plots') / experiment_name

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    f = 'FITNESS' if features_type == FeaturesType.FITNESS else 'PROBABILITIES'
    m = 'mean_' if mean else ''
    b = bandit if bandit else ''
    plt.savefig(plots_path / (bound + '_' + f + '_' + m + b + '_epochs_limit_' + str(epochs_limit) + '.pdf'))

    plt.show()
