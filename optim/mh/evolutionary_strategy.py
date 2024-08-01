import numpy as np

from config import Config
from optim.mh.metaheuristic import MetaHeuristic
from playground.environment import Environment
from utils import get_object_indices


class EvolutionaryStrategy(MetaHeuristic):
    def __init__(self, c, similarity_dict, locked_subsolution):
        super().__init__(c, similarity_dict, locked_subsolution)
        self.best_selection = None

    def strategy(self):
        population = self._get_initial_population()
        while self.count < self.c.MH_BUDGET:
            elites = (
                sorted(population, key=lambda x: x[1], reverse=True)[
                    : int(self.c.ES_ELITE_PROP * len(population))
                ]
            ).copy()
            population = self._crossover(population, len(elites))
            population = self._mutate(population)
            population.extend(elites)
            self.best_selection = self._get_best_selection(population)
        return self.best_selection

    def get_best_solution(self):
        return self.best_selection

    def _get_initial_population(self):
        population = []
        for _ in range(self.c.ES_POP_SIZE):
            s = self.get_random_selection()
            population.append((s, self.evaluate_selection(s)))
        return population

    def _mutate(self, population):
        to_mutate_indices = self._rng.choice(
            np.arange(len(population)),
            size=int(self.c.ES_MUTATION_RATE * len(population)),
            replace=False,
        )
        for i in to_mutate_indices:
            while True:
                one = self._rng.choice(np.where(population[i][0] == 1)[0])
                if one not in self.locked_subsolution:
                    break
            zero = self._rng.choice(np.where(population[i][0] == 0)[0])
            population[i][0][one] = 0
            population[i][0][zero] = 1
        return population

    def _crossover(self, population, num_elites):
        population = self._select_roulette(population, num_elites)
        self._rng.shuffle(population)
        new_population = []
        for i in range(0, len(population), 2):
            parent1, _ = population[i]
            parent1 = set(get_object_indices(parent1))
            parent2, _ = population[i + 1]
            parent2 = set(get_object_indices(parent2))
            child1 = np.zeros(self.c.OBJ_NUM)
            child2 = np.zeros(self.c.OBJ_NUM)
            # take the objects where both parents have the object
            common_objects = parent1 & parent2
            for obj in common_objects:
                child1[obj] = 1
                child2[obj] = 1
            different_objects = (parent1 - common_objects) | (parent2 - common_objects)
            # randomly select half of the objects that are different
            first_half = self._rng.choice(
                list(different_objects), size=len(different_objects) // 2, replace=False
            )
            second_half = different_objects - set(first_half)
            for obj in first_half:
                child1[obj] = 1
            for obj in second_half:
                child2[obj] = 1
            new_population.append((child1, self.evaluate_selection(child1)))
            new_population.append((child2, self.evaluate_selection(child2)))
        new_population = new_population[: len(population)]
        return new_population

    def _select_roulette(self, population, num_elites):
        scores = np.array([score for _, score in population])
        probabilities = scores / np.sum(scores)
        indices = np.arange(len(population))
        indices = self._rng.choice(
            indices,
            size=int(np.ceil((len(population) - num_elites) / 2) * 2),
            p=probabilities,
        )
        return [population[i] for i in indices]

    def _get_best_selection(self, population):
        best_selection = None
        best_score = 0
        for selection, score in population:
            assert (
                np.sum(selection) == self.c.DEMONSTRATION_BUDGET
            ), f"Selection does not have the correct number of objects: {selection}, {np.sum(selection)}"
            if score > best_score:
                best_score = score
                best_selection = selection
        return best_selection


if __name__ == "__main__":
    config = Config()
    env = Environment(config)
    es = EvolutionaryStrategy(config, env, [])

    selected = es.strategy()
    print(es.evaluate_selection_with_constraint_penalty(selected))
