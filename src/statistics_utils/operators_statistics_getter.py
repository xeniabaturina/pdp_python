import os

import pandas as pd

from pathlib import Path

from src.statistics_utils.statistics_plotter_utils import get_bounds_of_sub_experiment, get_sub_experiments_by_name, \
    get_experiment_parameters


def save_sub_experiment_result_to_csv(sub_experiment_name, sub_experiment_path, evol_algorithms, operators):
    bounds_with_dir_names = get_bounds_of_sub_experiment(sub_experiment_name, sub_experiment_path)

    dfs_results = {}

    for evol_algorithm in evol_algorithms:
        for (bound, bound_dir_name) in bounds_with_dir_names:
            for operator in operators:
                operator_df = pd.read_csv(sub_experiment_path / bound_dir_name / evol_algorithm / (operator + '.csv'), sep=',')

                print(evol_algorithm + " " + bound + " " + operator)

                if operator in dfs_results.keys():
                    dfs_results[operator] = pd.concat([dfs_results[operator], operator_df], axis=0)
                else:
                    dfs_results[operator] = operator_df

    experiments_results = {
        'operator': [operator for operator in operators],
        'f': [dfs_results[operator].is_succ.value_counts(normalize=True).loc[1] for operator in operators],
        't': [dfs_results[operator].time.mean() for operator in operators],
        'v': [max(0, dfs_results[operator].value.mean()) for operator in operators],
        'is_in_pool': [dfs_results[operator].is_in_pool.value_counts(normalize=True).loc[True]
                       for operator in operators],
        'was_best': [dfs_results[operator].was_best.value_counts(normalize=True).loc[True]
                     for operator in operators]
    }

    df_results_mean = pd.DataFrame(experiments_results)

    csv_files_path = '../../data/csv_files/'
    if not os.path.exists(csv_files_path):
        os.mkdir(csv_files_path)

    csv_path = csv_files_path + sub_experiment_name
    if not os.path.exists(csv_path):
        os.mkdir(csv_files_path + sub_experiment_name)

    df_results_mean.to_csv(csv_path + '/operators_statistics.csv', index=False)


def save_experiment_result_to_csv(experiment_name):
    strategies, evol_algorithms, operators = get_experiment_parameters(experiment_name)
    sub_experiments_names = get_sub_experiments_by_name(experiment_name)

    operators_logs_path = Path('../../data/operators_logs/')

    for sub_experiment_name in sub_experiments_names:
        sub_experiment_path = operators_logs_path / sub_experiment_name

        save_sub_experiment_result_to_csv(sub_experiment_name, sub_experiment_path, evol_algorithms, operators)


if __name__ == "__main__":
    save_experiment_result_to_csv('for_statistics')
