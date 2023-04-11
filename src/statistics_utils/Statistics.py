from typing import List, AnyStr

import pandas as pd


class Statistics(object):
    def __init__(self, experiment_name: AnyStr, operators: List['Operator'], csv_files_path):
        self.experiment_name = experiment_name
        self.operators = operators

        csv_path = csv_files_path + experiment_name

        self.statistics = pd.read_csv(csv_path + '/operators_statistics.csv', sep=',', index_col=0)

    def operator_statistics(self, operator_id):
        statistics = {
            'f': self.statistics.iloc[operator_id]['f'],
            't': self.statistics.iloc[operator_id]['t'],
            'v': self.statistics.iloc[operator_id]['v'],
            'is_in_pool': self.statistics.iloc[operator_id]['is_in_pool'],
            'was_best': self.statistics.iloc[operator_id]['was_best']
        }

        return statistics
