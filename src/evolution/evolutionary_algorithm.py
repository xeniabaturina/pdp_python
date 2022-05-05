from abc import ABC, abstractmethod

from ..graph_utils.Graph import Graph, Mode


class Individual(ABC):
    def __init__(self, g: Graph, distances):
        self.g = g
        self.g.set_path_weight(g.get_path_weight(distances))

    @abstractmethod
    def mutate(self, mutation, distances, bandit):
        pass


class TSP(Individual):
    def mutate(self, mutation, distances, bandit):
        try:
            choice = bandit.policy.choice() if bandit else None
        except Exception as e:
            choice = 1

        g_mutated = mutation(self.g, distances, choice) if bandit else mutation(self.g, distances)

        new_g = Graph(Mode.EDGES, g_mutated.edges, self.g.dataset_dir)
        new_g.set_path_weight(g_mutated.get_path_weight(distances))
        new_g.coordinates_from_csv = self.g.coordinates_from_csv

        old_weight = self.g.get_path_weight(distances)
        new_weight = new_g.get_path_weight(distances)

        if bandit:
            r = (1 - new_weight / old_weight) * 100
            reward = bandit.env.draw(choice, r, bandit.time)
            l = (new_weight / old_weight) * 100
            bandit.policy.get_reward(choice, l, reward)
            bandit.results.store(bandit.time, choice, reward)

        if new_weight < old_weight:
            if new_g.is_valid():
                self.g = new_g


class Population:
    def __init__(self, size, fitness, individual_class, initial_graph, distances):
        self.initial_graph = initial_graph
        self.distances = distances
        self.initial_graph.set_path_weight(initial_graph.get_path_weight(self.distances))
        self.fitness = fitness
        self.individuals = [individual_class(self.initial_graph, self.distances) for _ in range(size)]

        self.sort_individuals()

    def replace(self, new_individuals):
        size = len(self.individuals)
        self.individuals.extend(new_individuals)
        self.sort_individuals()
        self.individuals = self.individuals[-size:]

    def sort_individuals(self):
        for individual in self.individuals:
            individual.g.set_path_weight(individual.g.get_path_weight(self.distances))

        self.individuals.sort(key=lambda x: fitness(x, self.distances))

    def get_individuals(self):
        return self.individuals


class Evolution:
    def __init__(self, pool_size, fitness, individual_class, mutation, initial_graph, distances, bandit=None):
        self.distances = distances
        self.mutation = mutation
        self.bandit = bandit
        self.pool = Population(pool_size, fitness, individual_class, initial_graph, self.distances)

    def step(self):
        individuals = self.pool.get_individuals()
        offsprings = []

        for ind in individuals:
            for _ in range(5):
                offspring = ind
                offspring.mutate(self.mutation, self.distances, self.bandit)
                if self.bandit:
                    self.bandit.time_inc()
                offsprings.append(offspring)

        self.pool.replace(offsprings)

    def get_best_g(self):
        return self.pool.individuals[-1].g

    def get_best_path(self):
        return self.get_best_g().optimize()

    def get_best_weight(self):
        return self.get_best_g().get_path_weight(self.distances)


def fitness(individual, distances):
    return -individual.g.get_path_weight(distances)
