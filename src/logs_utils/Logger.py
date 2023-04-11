import csv
import os
import time


class Logger(object):
    def __init__(self, orders_num, epochs_num, pool_size, experiment_logs_path=None, operators_logs_path=None,
                 dataset_path=None, inner_datasets_num=None, inner_experiments_num=None):
        self.orders_num = orders_num
        self.epochs_num = epochs_num
        self.pool_size = pool_size
        self.experiment_logs_path = experiment_logs_path
        self.operators_logs_path = operators_logs_path
        self.dataset_path = dataset_path
        self.inner_datasets_num = inner_datasets_num
        self.inner_experiments_num = inner_experiments_num

        self.delayed_operator_statistics = []
        self.best_pool_weight = 0

        self.operators_names_lst = []
        self.strategies_names_lst = []

    @staticmethod
    def update_operator_logs_file(operator_logs_path):
        if not operator_logs_path.is_file():
            os.makedirs(os.path.dirname(operator_logs_path), exist_ok=True)

            with open(operator_logs_path, 'w') as f:
                writer = csv.writer(f)

                header = ['orders_num', 'n_ep', 'pool']
                header += ['is_succ', 'value', 'time', 'is_in_pool', 'was_best']

                writer.writerow(header)

    def log_operator_statistics(self, max_pool_weight):
        for (operator_short_name, is_succ, improvement_pc, elapsed_time_ms, final_weight, was_best) \
                in self.delayed_operator_statistics:
            operator_logs_path = self.operators_logs_path / f'{operator_short_name}.csv'
            self.update_operator_logs_file(operator_logs_path)

            with open(operator_logs_path, 'a') as f:
                writer = csv.writer(f)

                row = [self.orders_num, self.epochs_num, self.pool_size]
                row.extend([is_succ, improvement_pc, elapsed_time_ms, final_weight <= max_pool_weight, was_best])

                writer.writerow(row)

        self.delayed_operator_statistics = []

    def update_best_pool_weight(self, new_best_pool_weight):
        self.best_pool_weight = new_best_pool_weight

    def check_best_pool_weight(self, weight):
        self.best_pool_weight = min(self.best_pool_weight, weight)

    def delay_operator_statistics(self, operator_short_name, is_succ, improvement_pc, elapsed_time_ms, final_weight):
        self.delayed_operator_statistics.append([operator_short_name, is_succ, improvement_pc, elapsed_time_ms,
                                                 final_weight, final_weight <= self.best_pool_weight])

    def update_inner_experiment_logs_file(self, inner_experiment_id):
        inner_experiment_path = self.experiment_logs_path / f'inner_experiment_{inner_experiment_id}.csv'

        if not inner_experiment_path.is_file():
            os.makedirs(os.path.dirname(inner_experiment_path), exist_ok=True)

            with open(inner_experiment_path, 'w+') as f:
                writer = csv.writer(f)

                header = ['simulation_id', 'inner_dataset', 'orders_num', 'n_ep', 'pool', 'start_length']
                suffixes = ['progress', 'time', 'epochs']
                for operator_name in self.operators_names_lst:
                    suffixes.append(operator_name + '_pulls')

                for operator_name in self.operators_names_lst:
                    suffixes.append(operator_name + '_pulls_succ')

                for strategy_name in self.strategies_names_lst:
                    header += [f'{strategy_name}_{s}' for s in suffixes]

                writer.writerow(header)

    def log_inner_experiment(self, inner_experiment_id, simulation_id, evolutions,
                             inner_dataset, epochs_num, pool_size, start_length):
        inner_experiment_path = self.experiment_logs_path / f'inner_experiment_{inner_experiment_id}.csv'

        with open(inner_experiment_path, 'a') as f:
            writer = csv.writer(f)
            row = [simulation_id, inner_dataset, self.orders_num, epochs_num, pool_size, start_length]
            for evolution in evolutions:
                row.extend([evolution.progress, evolution.time, evolution.epochs,
                            *evolution.pulls, *evolution.pulls_succ])

            writer.writerow(row)

    def update_inner_experiment_details_logs_files(self, inner_experiment_id):
        inner_experiment_details_path = (self.experiment_logs_path / f'details/inner_experiment_{inner_experiment_id}')

        for i, strategy_name in enumerate(self.strategies_names_lst):
            with open(inner_experiment_details_path / f'{strategy_name}.csv', 'w+') as f:
                writer = csv.writer(f)
                header = ['simulation_id', 'fitness_hist', 'time_hist', 'probabilities']
                writer.writerow(header)

    def log_inner_experiment_details(self, inner_experiment_id, simulation_id, evolutions):
        inner_experiment_details_path = (self.experiment_logs_path / f'details/inner_experiment_{inner_experiment_id}')

        for i, strategy_name in enumerate(self.strategies_names_lst):
            with open(inner_experiment_details_path / f'{strategy_name}.csv', 'a') as f:
                writer = csv.writer(f)
                row = [simulation_id,
                       '#'.join(str(x) for x in (evolutions[i].get_fitness_hist())),
                       '#'.join(str(x) for x in (evolutions[i].get_time_hist()))]

                probabilities = evolutions[i].get_probabilities()

                row.append('&' + '#'.join(str(x) for x in probabilities))
                writer.writerow(row)

    def rewrite_coordinates_file(self, coordinates):
        self.experiment_logs_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        for inner_experiment in range(1, self.inner_experiments_num + 1):
            (self.experiment_logs_path / f'details/inner_experiment_{inner_experiment}').mkdir(parents=True,
                                                                                               exist_ok=True)

        coordinates_file = (self.dataset_path / f'coordinates_by_id.csv')

        if coordinates is not None:
            with open(coordinates_file, 'w+') as f:
                writer = csv.writer(f)
                header = ['coord_x', 'coord_y']
                writer.writerow(header)

                for coord_x, coord_y in coordinates:
                    next_row = [coord_x, coord_y]

                    writer.writerow(next_row)

    def rewrite_orders_files(self, orders):
        for inner_dataset in range(1, self.inner_datasets_num + 1):
            (self.dataset_path / f'dataset_{inner_dataset}').mkdir(parents=True, exist_ok=True)

            orders_path = (self.dataset_path / f'dataset_{inner_dataset}/orders.csv')

            if orders is not None:
                with open(orders_path, 'w+') as f:
                    writer = csv.writer(f)
                    header = ['from', 'to']
                    writer.writerow(header)

                    for order_from, order_to in orders:
                        next_row = [order_from, order_to]

                        writer.writerow(next_row)
