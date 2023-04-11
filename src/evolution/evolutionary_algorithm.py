from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from ..graph_utils.Graph import Graph


class Individual(ABC):
    def __init__(self, g: Graph, calculated_weights):
        self.g = g
        self.g.set_path_weight(g.get_path_weight(calculated_weights))

    @abstractmethod
    def mutate(self, calculated_weights, bandit):
        pass


class TSP(Individual):
    def mutate(self, calculated_weights, strategy):
        operator_id = strategy.choose_operator_id()

        old_weight = self.g.get_path_weight(calculated_weights)

        g_new = strategy.apply_operator(operator_id, self.g, calculated_weights)

        new_weight = g_new.get_path_weight(calculated_weights)

        if new_weight < old_weight:
            if g_new.is_valid():
                self.g = g_new


class Population:
    def __init__(self, size, fitness, individual_class, initial_graph, calculated_weights, logger):
        self.initial_graph = initial_graph
        self.calculated_weights = calculated_weights
        self.initial_graph.set_path_weight(initial_graph.get_path_weight(self.calculated_weights))
        self.fitness = fitness
        self.individuals = [individual_class(self.initial_graph, self.calculated_weights) for _ in range(size)]
        self.logger = logger

        self.sort_individuals()

    def replace(self, new_individuals):
        size = len(self.individuals)

        self.individuals.extend(new_individuals)
        self.sort_individuals()
        self.individuals = self.individuals[-size:]

        self.logger.log_operator_statistics(self.individuals[0].g.get_path_weight(self.calculated_weights))

    def sort_individuals(self):
        for individual in self.individuals:
            individual.g.set_path_weight(individual.g.get_path_weight(self.calculated_weights))

        self.individuals.sort(key=lambda x: fitness(x, self.calculated_weights))

    def get_individuals(self):
        return self.individuals


class Evolution:
    def __init__(self, pool_size, offsprings_num, fitness, evolution_type, individual_class, initial_graph, calculated_weights, strategy, logger):
        self.evolution_type = evolution_type
        self.calculated_weights = calculated_weights
        self.strategy = strategy
        self.offsprings_num = offsprings_num
        self.logger = logger
        self.pool = Population(pool_size, fitness, individual_class, initial_graph, self.calculated_weights, self.logger)

    def step(self):
        individuals = self.pool.get_individuals()
        offsprings = []
        self.logger.update_best_pool_weight(np.Inf)

        for ind in individuals:
            for _ in range(self.offsprings_num):
                offspring = deepcopy(ind)  # toDo: delete this
                offspring.mutate(self.calculated_weights, self.strategy)
                self.logger.check_best_pool_weight(offspring.g.get_path_weight(self.calculated_weights))
                offsprings.append(offspring)
            if self.evolution_type.value == 'K+KN':
                offspring = deepcopy(ind)  # toDo: delete this
                offsprings.append(offspring)

        self.pool.replace(offsprings)

    def get_best_g(self):
        return self.pool.individuals[-1].g

    def get_best_path(self):
        return self.get_best_g().optimize()

    def get_best_weight(self):
        return self.get_best_g().get_path_weight(self.calculated_weights)


def fitness(individual, calculated_weights):
    return -individual.g.get_path_weight(calculated_weights)
