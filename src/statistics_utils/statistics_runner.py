from src.statistics_utils.statistics_getter_utils import save_experiment_result_to_csv
from src.statistics_utils.statistics_plotter_utils import plot_features_by_experiment_and_bound, FeaturesType, \
    get_experiment_parameters


def plot_fitness(experiment_name, mean):
    plot_features_by_experiment_and_bound(FeaturesType.FITNESS, experiment_name, bound, mean=mean, alpha=0.5 if mean else 0.2)


def plot_probabilities(experiment_name):
    strategies, evol_algorithms, operators = get_experiment_parameters(experiment_name)
    for bandit in strategies:
        plot_features_by_experiment_and_bound(FeaturesType.PROBABILITIES, experiment_name, bound, True, bandit, False)


if __name__ == "__main__":
    bound = 'inf'
    for experiment_name in ['test_all', 'test_all_no_rb', 'test_all_new_operators']:
        save_experiment_result_to_csv(experiment_name)
        plot_fitness(experiment_name, True)
        plot_probabilities(experiment_name)
