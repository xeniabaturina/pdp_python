import os

import pandas as pd

from pathlib import Path

from src.statistics_utils.statistics_plotter_utils import get_bounds_of_sub_experiment, get_sub_experiments_by_name, \
    get_experiment_parameters


def save_sub_experiment_result_to_csv(sub_experiment_name, sub_experiment_path, strategies, evol_algorithms):
    bounds_with_dir_names = get_bounds_of_sub_experiment(sub_experiment_name, sub_experiment_path)

    dfs_results_by_bound = {}

    for (bound, _) in bounds_with_dir_names:
        dfs_results_by_bound[bound] = {}

    for evol_algorithm in evol_algorithms:
        for (bound, bound_dir_name) in bounds_with_dir_names:

            logs_path = sub_experiment_path / bound_dir_name / evol_algorithm

            simulations_paths = [x for x in logs_path.glob('*') if x.is_file()]

            list_dfs = []
            for inner_experiment_path in simulations_paths:
                df = pd.read_csv(inner_experiment_path, sep=',')

                print(df['pure-relocate_block_progress'])

                list_dfs.append(df)

            dfs_results_by_bound[bound][evol_algorithm] = list_dfs

    for (bound, _) in bounds_with_dir_names:
        experiments_results = {}

        for evol_algorithm in evol_algorithms:
            if evol_algorithm in dfs_results_by_bound[bound]:
                df = pd.concat(dfs_results_by_bound[bound][evol_algorithm], ignore_index=True)

                for col in df.columns:
                    print(col)
                experiments_results[evol_algorithm] = [df[strategy + '_progress'].mean() for strategy in strategies]

        df_results_mean = pd.DataFrame(experiments_results, index=strategies)

        bound_dir_name = [y for (x, y) in bounds_with_dir_names if x == bound][0]

        csv_files_path = '../../data/csv_files/'
        if not os.path.exists(csv_files_path):
            os.mkdir(csv_files_path)

        csv_path = csv_files_path + sub_experiment_name + '/' + bound_dir_name
        if not os.path.exists(csv_path):
            if not os.path.exists(csv_files_path + sub_experiment_name):
                os.mkdir(csv_files_path + sub_experiment_name)
            os.mkdir(csv_files_path + sub_experiment_name + '/' + bound_dir_name)

        df_results_mean.to_csv(csv_path + '/results_mean.csv', index=False)


def save_mean_result_of_experiment(experiment_name, strategies, evol_algorithms):
    sub_experiments_names = get_sub_experiments_by_name(experiment_name)

    experiments_results_path = Path('../../data/csv_files')

    dfs_results = []

    for sub_experiment_name in sub_experiments_names:
        sub_experiment_path = experiments_results_path / sub_experiment_name

        bounds_with_dir_names = get_bounds_of_sub_experiment(sub_experiment_name, sub_experiment_path)

        mux = pd.MultiIndex.from_product([(bound for (bound, _) in bounds_with_dir_names), evol_algorithms], names=['bound', 'evol_algorithm'])

        data = [[] for _ in range(len(strategies))]

        for (bound, bound_dir_name) in bounds_with_dir_names:
            df_result = pd.read_csv(sub_experiment_path / bound_dir_name / 'results_mean.csv', sep=',')

            for evol_algorithm in evol_algorithms:
                for i, value in enumerate(df_result[evol_algorithm].tolist()):
                    data[i].append(value)

        df_result = pd.DataFrame(data, columns=mux, index=strategies)
        dfs_results.append(df_result)

    dfs_results_concat = pd.concat(dfs_results)
    by_row_index = dfs_results_concat.groupby(dfs_results_concat.index)
    df_results_mean = by_row_index.mean()

    if not os.path.exists(experiments_results_path / experiment_name):
        os.mkdir(experiments_results_path / experiment_name)

    df_results_mean.to_csv(experiments_results_path / experiment_name / 'results_mean.csv', index=strategies)


def save_experiment_result_to_csv(experiment_name):
    strategies, evol_algorithms, operators = get_experiment_parameters(experiment_name)

    sub_experiments_names = get_sub_experiments_by_name(experiment_name)

    experiments_logs_path = Path('../../data/experiments_logs/')

    for sub_experiment_name in sub_experiments_names:
        sub_experiment_path = experiments_logs_path / sub_experiment_name

        save_sub_experiment_result_to_csv(sub_experiment_name, sub_experiment_path, strategies, evol_algorithms)

    save_mean_result_of_experiment(experiment_name, strategies, evol_algorithms)
