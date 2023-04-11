import itertools
import time

from multiprocessing import Process
from pathlib import Path
from enum import Enum

from src.evolution.run_evolution import run_evolution
from src.experiments.experiments_data_generator import generate_orders, OrdersModelType, generate_coordinates
from src.graph_utils.Graph import Graph, Mode
from src.logs_utils.Logger import Logger

from src.operators_utils.operators import Lin2Opt, DoubleBridge, PointExchange, CoupleExchange, RelocateBlock, \
    RandomRelocateBlock, Combined
from src.statistics_utils.Statistics import Statistics
from src.strategies_utils.strategies import EpsilonGreedy, UCB1, Softmax, Thompson, Pure, S_v, S_f, S_pool, S_best, S_n, \
    S_fn, Random

all_evol_algorithms = ('1+1', '1,1', '1+N', '1,N', 'K+KN', 'K,KN')


class EvolutionType(Enum):
    PARENTS_LIVE = 'K+KN'
    PARENTS_DIE = 'K,KN'


class EvolutionaryAlgorithm:
    def __init__(self, name, evolution_type, epochs_num, pool_size, offsprings_num):
        self.name = name
        self.evolution_type = evolution_type
        self.epochs_num = epochs_num
        self.pool_size = pool_size
        self.offsprings_num = offsprings_num


class OrdersModel:
    def __init__(self, model_type, key_points_num, districts_num):
        self.type = model_type
        self.key_points_num = key_points_num
        self.districts_num = districts_num


class Experiment:
    def __init__(self, logger, name, strategies_lst, inner_experiments_num, inner_datasets_num, orders_num, turns_num, orders_model,
                 evolutionary_algorithm, derivative_bound=None, dataset_name=None, dataset_recreate=True, gaussian=True,
                 tqdm_disable=False, general_coordinates=None, general_orders=None):
        self.general_coordinates = general_coordinates
        self.general_orders = general_orders
        self.name = name

        self.strategies_lst = strategies_lst

        self.inner_experiments_num = inner_experiments_num
        self.inner_datasets_num = inner_datasets_num
        self.orders_num = orders_num
        self.turns_num = turns_num

        self.orders_model = orders_model
        self.evolutionary_algorithm = evolutionary_algorithm
        self.derivative_bound = derivative_bound
        self.tqdm_disable = tqdm_disable

        self.dataset_name = dataset_name
        if self.dataset_name is None:
            self.dataset_name = self.name

        self.dataset_recreate = dataset_recreate
        self.gaussian = gaussian

        self.experiment_logs_path = Path('../../../pdp_python/data/experiments_logs/' + self.name + '/' + evolutionary_algorithm.name)
        self.operators_logs_path = Path('../../../pdp_python/data/operators_logs/' + self.name + '/' + evolutionary_algorithm.name)
        self.dataset_path = Path('../../../pdp_python/data/datasets/' + self.dataset_name)

        logger.experiment_logs_path = self.experiment_logs_path
        logger.operators_logs_path = self.operators_logs_path
        logger.dataset_path = self.dataset_path
        logger.inner_datasets_num = inner_datasets_num
        logger.inner_experiments_num = inner_experiments_num

        self.logger = logger

    def run(self):
        self.logger.rewrite_coordinates_file(self.general_coordinates)
        self.logger.rewrite_orders_files(self.general_orders)

        for inner_experiment_id in range(1, self.inner_experiments_num + 1):
            print(f'\n({self.derivative_bound}_{self.evolutionary_algorithm.name}) INNER EXPERIMENT: {inner_experiment_id} / {self.inner_experiments_num} -=======================- ')

            self.logger.update_inner_experiment_logs_file(inner_experiment_id)
            self.logger.update_inner_experiment_details_logs_files(inner_experiment_id)

            simulation_id = 0
            for inner_dataset in range(1, self.inner_datasets_num + 1):
                for turn in range(1, self.turns_num + 1):
                    simulation_id += 1

                    log = f'Dataset {inner_dataset} / {self.inner_datasets_num} --- turn: {turn} / {self.turns_num}'

                    g = Graph(Mode.IDS, dataset_dir=str(self.dataset_path / f'dataset_{inner_dataset}'))
                    g.set_path_weight(g.get_path_weight())
                    start_length = g.get_path_weight()

                    epochs_num = self.evolutionary_algorithm.epochs_num
                    pool_size = self.evolutionary_algorithm.pool_size
                    offsprings_num = self.evolutionary_algorithm.offsprings_num
                    evolution_type = self.evolutionary_algorithm.evolution_type
                    derivative_bound = self.derivative_bound
                    tqdm_disable = self.tqdm_disable

                    evolutions = []

                    for strategy in self.strategies_lst:
                        strategy.setup()
                        evolution = run_evolution(logger=self.logger,
                                                  n_epochs=epochs_num, pool_size=pool_size,
                                                  offsprings_num=offsprings_num,
                                                  initial_graph=g, strategy=strategy,
                                                  derivative_bound=derivative_bound,
                                                  evolution_type=evolution_type, log=log, tqdm_disable=tqdm_disable)

                        evolutions.append(evolution)

                    self.logger.log_inner_experiment(inner_experiment_id, simulation_id, evolutions,
                                                     inner_dataset, epochs_num, pool_size, start_length)
                    self.logger.log_inner_experiment_details(inner_experiment_id, simulation_id, evolutions)


def set_experiment(logger, name, general_name, strategies_lst, epochs_num, pool_size, offsprings_num, evolution_type,
                   model_type=OrdersModelType.MANY_TO_ONE, derivative_bound=0,
                   inner_experiments_num=10, inner_datasets_num=5,
                   key_points_num=None, districts_num=1,
                   orders_num=50, turns_num=1,
                   dataset_recreate=False, gaussian=True, tqdm_disable=False, general_coordinates=None, general_orders=None):
    evolutionary_algorithm = EvolutionaryAlgorithm(name=name, evolution_type=evolution_type,
                                                   epochs_num=epochs_num, pool_size=pool_size, offsprings_num=offsprings_num)

    orders_model = OrdersModel(model_type=model_type, key_points_num=key_points_num, districts_num=districts_num)

    experiment = Experiment(logger=logger, name=general_name, strategies_lst=strategies_lst, inner_experiments_num=inner_experiments_num,
                            inner_datasets_num=inner_datasets_num, orders_num=orders_num, turns_num=turns_num,
                            orders_model=orders_model,
                            evolutionary_algorithm=evolutionary_algorithm,
                            derivative_bound=derivative_bound,
                            dataset_recreate=dataset_recreate, gaussian=gaussian, tqdm_disable=tqdm_disable,
                            general_coordinates=general_coordinates, general_orders=general_orders)
    experiment.run()


def set_all_experiments_parallel(logger, experiment_name, strategies_lst, gaussian, epochs_num, max_pool_size, max_offsprings_num, derivative_bound,
                                 inner_experiments_num, inner_datasets_num, general_coordinates=None, general_orders=None):
    data_type = 'gaussian' if gaussian else 'taxi'
    derivative_bound_str = format(derivative_bound, ".1E") if derivative_bound > 0 else "inf"
    general_name = f'{experiment_name}/{experiment_name}_{epochs_num}_{max_pool_size}_{derivative_bound_str}_{data_type}'


    evolutionary_algorithms = {"1+1": {"pool_size": 1, "offsprings_num": 1, "evolution_type": EvolutionType.PARENTS_LIVE},
                               "1,1": {"pool_size": 1, "offsprings_num": 1, "evolution_type": EvolutionType.PARENTS_DIE},
                               "1+N": {"pool_size": 1, "offsprings_num": max_offsprings_num, "evolution_type": EvolutionType.PARENTS_LIVE},
                               "1,N": {"pool_size": 1, "offsprings_num": max_offsprings_num, "evolution_type": EvolutionType.PARENTS_DIE},
                               "K+KN": {"pool_size": max_pool_size, "offsprings_num": max_offsprings_num, "evolution_type": EvolutionType.PARENTS_LIVE},
                               "K,KN": {"pool_size": max_pool_size, "offsprings_num": max_offsprings_num, "evolution_type": EvolutionType.PARENTS_DIE}}

    processes = []

    for evolutionary_algorithm_name in all_evol_algorithms:
        pool_size = evolutionary_algorithms[evolutionary_algorithm_name]["pool_size"]
        offsprings_num = evolutionary_algorithms[evolutionary_algorithm_name]["offsprings_num"]
        evolution_type = evolutionary_algorithms[evolutionary_algorithm_name]["evolution_type"]

        process = Process(target=set_experiment, kwargs={"logger": logger,
                                                         "name": evolutionary_algorithm_name,
                                                         "general_name": general_name,
                                                         "strategies_lst": strategies_lst,
                                                         "epochs_num": epochs_num,
                                                         "pool_size": pool_size,
                                                         "offsprings_num": offsprings_num,
                                                         "evolution_type": evolution_type,
                                                         "derivative_bound": derivative_bound,
                                                         "inner_experiments_num": inner_experiments_num,
                                                         "inner_datasets_num": inner_datasets_num,
                                                         "tqdm_disable": False,
                                                         "general_coordinates": general_coordinates,
                                                         "general_orders": general_orders})
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def test_all(points_num, coordinates=None):
    orders_num = 50
    orders_model_type = OrdersModelType.NO_K_POINTS

    epochs_num = 250
    max_pool_size = 50
    max_offsprings_num = 25

    logger = Logger(orders_num, epochs_num, max_pool_size)

    simple_operators_lst = [Lin2Opt(logger), DoubleBridge(logger),
                            PointExchange(logger), CoupleExchange(logger),
                            RelocateBlock(logger)]

    for simple_operator in simple_operators_lst:
        print(simple_operator)

    operators_lst = simple_operators_lst

    strategies_lst = [
        EpsilonGreedy(operators_lst, epsilon=0.6),
        UCB1(operators_lst),
        Softmax(operators_lst, tau=3),
        Thompson(operators_lst),
        S_n(operators_lst),
        S_fn(operators_lst),
        Random(operators_lst),
        Pure(Lin2Opt(logger)),
        Pure(RelocateBlock(logger))
    ]

    logger.operators_names_lst = [operator.short_name for operator in operators_lst]
    logger.strategies_names_lst = [strategy.short_name for strategy in strategies_lst]

    orders = generate_orders(orders_model_type, points_num, orders_num)
    for derivative_bound in [0]:
        print("DERIVATIVE BOUND IS ", derivative_bound)
        set_all_experiments_parallel(logger, experiment_name=f"test_all",
                                     strategies_lst=strategies_lst,
                                     gaussian=True, epochs_num=epochs_num, max_pool_size=max_pool_size,
                                     max_offsprings_num=max_offsprings_num,
                                     inner_experiments_num=5, inner_datasets_num=3,
                                     derivative_bound=derivative_bound,
                                     general_coordinates=coordinates, general_orders=orders
                                     )


def test_all_no_rb(points_num, coordinates=None):
    orders_num = 50
    orders_model_type = OrdersModelType.NO_K_POINTS

    epochs_num = 250
    max_pool_size = 50
    max_offsprings_num = 25

    logger = Logger(orders_num, epochs_num, max_pool_size)

    simple_operators_lst = [Lin2Opt(logger), DoubleBridge(logger),
                            PointExchange(logger), CoupleExchange(logger)]

    for simple_operator in simple_operators_lst:
        print(simple_operator)

    operators_lst = simple_operators_lst

    strategies_lst = [
        EpsilonGreedy(operators_lst, epsilon=0.6),
        UCB1(operators_lst),
        Softmax(operators_lst, tau=3),
        Thompson(operators_lst),
        S_n(operators_lst),
        S_fn(operators_lst),
        Random(operators_lst),
        Pure(Lin2Opt(logger)),
        Pure(RelocateBlock(logger))
    ]

    logger.operators_names_lst = [operator.short_name for operator in operators_lst]
    logger.strategies_names_lst = [strategy.short_name for strategy in strategies_lst]

    orders = generate_orders(orders_model_type, points_num, orders_num)
    for derivative_bound in [0]:
        print("DERIVATIVE BOUND IS ", derivative_bound)
        set_all_experiments_parallel(logger, experiment_name=f"test_all_no_rb",
                                     strategies_lst=strategies_lst,
                                     gaussian=True, epochs_num=epochs_num, max_pool_size=max_pool_size,
                                     max_offsprings_num=max_offsprings_num,
                                     inner_experiments_num=5, inner_datasets_num=3,
                                     derivative_bound=derivative_bound,
                                     general_coordinates=coordinates, general_orders=orders
                                     )


def test_all_new_operators(points_num, coordinates=None):
    orders_num = 50
    orders_model_type = OrdersModelType.NO_K_POINTS

    epochs_num = 250
    max_pool_size = 50
    max_offsprings_num = 25

    logger = Logger(orders_num, epochs_num, max_pool_size)

    simple_operators_lst = [Lin2Opt(logger), DoubleBridge(logger),
                            PointExchange(logger), CoupleExchange(logger),
                            RelocateBlock(logger), RandomRelocateBlock(logger)]

    operators_lst = simple_operators_lst

    for (op1, op2) in itertools.product(simple_operators_lst, simple_operators_lst):
        print(op1, op2)
        operators_lst.append(Combined(logger, op1, op2))

    strategies_lst = [
        EpsilonGreedy(operators_lst, epsilon=0.6),
        UCB1(operators_lst),
        Softmax(operators_lst, tau=3),
        Thompson(operators_lst),
        Random(operators_lst),
        Pure(Lin2Opt(logger)),
        Pure(RelocateBlock(logger))
    ]

    logger.operators_names_lst = [operator.short_name for operator in operators_lst]
    logger.strategies_names_lst = [strategy.short_name for strategy in strategies_lst]

    orders = generate_orders(orders_model_type, points_num, orders_num)
    for derivative_bound in [0]:
        print("DERIVATIVE BOUND IS ", derivative_bound)
        set_all_experiments_parallel(logger, experiment_name=f"test_all_new_operators",
                                     strategies_lst=strategies_lst,
                                     gaussian=True, epochs_num=epochs_num, max_pool_size=max_pool_size,
                                     max_offsprings_num=max_offsprings_num,
                                     inner_experiments_num=5, inner_datasets_num=3,
                                     derivative_bound=derivative_bound,
                                     general_coordinates=coordinates, general_orders=orders
                                     )


def for_statistics(points_num, coordinates=None):
    orders_num = 100
    orders_model_type = OrdersModelType.NO_K_POINTS

    epochs_num = 250
    max_pool_size = 25
    max_offsprings_num = 5

    logger = Logger(orders_num, epochs_num, max_pool_size)

    simple_operators_lst = [Lin2Opt(logger), DoubleBridge(logger),
                            PointExchange(logger), CoupleExchange(logger),
                            RelocateBlock(logger), RandomRelocateBlock(logger)]

    for simple_operator in simple_operators_lst:
        print(simple_operator)

    operators_lst = simple_operators_lst

    for (op1, op2) in itertools.product(simple_operators_lst, simple_operators_lst):
        print(op1, op2)
        operators_lst.append(Combined(logger, op1, op2))

    strategies_lst = [
        Pure(operator) for operator in operators_lst
    ]

    # orders = generate_orders(orders_model_type, points_num, orders_num)
    for derivative_bound in [0]:
        print("DERIVATIVE BOUND IS ", derivative_bound)
        set_all_experiments_parallel(logger, experiment_name=f"for_statistics",
                                     strategies_lst=strategies_lst,
                                     gaussian=True, epochs_num=epochs_num, max_pool_size=max_pool_size,
                                     max_offsprings_num=max_offsprings_num,
                                     inner_experiments_num=1, inner_datasets_num=1, derivative_bound=derivative_bound,
                                     # general_coordinates=coordinates, general_orders=orders
                                     )


if __name__ == "__main__":
    start_time = time.time()

    points_num = 300 * 3

    gaussian = True
    # coordinates = generate_coordinates(gaussian, points_num)

    test_all(points_num)

    print("--- %s seconds ---" % (time.time() - start_time))
