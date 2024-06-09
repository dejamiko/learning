import numpy as np

from al.utils import MetaHeuristic, get_object_indices
from config import Config


class EvolutionaryStrategy(MetaHeuristic):
    def __init__(self, c, threshold=None):
        super().__init__(c, threshold)

    def strategy(self):
        population = self._get_initial_population()
        for k in range(self.c.ES_ITER):
            elites = (sorted(population, key=lambda x: x[1], reverse=True)[:int(self.c.ES_ELITE_PROP * len(population))]).copy()
            population = self._crossover(population, len(elites))
            population = self._mutate(population)
            population.extend(elites)
            self.best_selection = self._get_best_selection(population)
        return self.best_selection

    def _get_initial_population(self):
        population = []
        for _ in range(self.c.ES_POP_SIZE):
            s = self._get_initial_selection()
            population.append((s, self.evaluate_selection(s)))
        return population

    def _mutate(self, population):
        to_mutate_indices = np.random.choice(
            np.arange(len(population)),
            size=int(self.c.ES_MUTATION_RATE * len(population)),
            replace=False,
        )
        for i in to_mutate_indices:
            one = np.random.choice(np.where(population[i][0] == 1)[0])
            zero = np.random.choice(np.where(population[i][0] == 0)[0])
            population[i][0][one] = 0
            population[i][0][zero] = 1
        return population

    def _crossover(self, population, num_elites):
        population = self._select_roulette(population, num_elites)
        np.random.shuffle(population)
        new_population = []
        for i in range(0, len(population), 2):
            parent1, _ = population[i]
            parent1 = get_object_indices(parent1)
            parent2, _ = population[i + 1]
            parent2 = get_object_indices(parent2)
            child1 = np.zeros(self.c.OBJ_NUM)
            child2 = np.zeros(self.c.OBJ_NUM)
            for p1, p2 in zip(parent1, parent2):
                if np.random.uniform() < 0.5:
                    child1[p1] = 1
                    child2[p2] = 1
                else:
                    child1[p2] = 1
                    child2[p1] = 1
            new_population.append((child1, self.evaluate_selection(child1)))
            new_population.append((child2, self.evaluate_selection(child2)))
        new_population = new_population[: len(population)]
        return new_population

    def _select_roulette(self, population, num_elites):
        scores = [score for _, score in population]
        probabilities = scores / np.sum(scores)
        assert np.isclose(np.sum(probabilities), 1), f"Probabilities do not sum to 1: {np.sum(probabilities)}"
        indices = np.arange(len(population))
        indices = np.random.choice(indices, size=int(np.ceil((len(population) - num_elites) / 2) * 2), p=probabilities)
        return [population[i] for i in indices]

    def _get_best_selection(self, population):
        best_selection = None
        best_score = 0
        for selection, score in population:
            if score > best_score:
                best_score = score
                best_selection = selection
        return best_selection


if __name__ == "__main__":
    c = Config()
    es = EvolutionaryStrategy(c)

    results = []
    for pop_size in np.linspace(50, 300, 5):
        for mutation_rate in np.linspace(0.01, 0.3, 5):
            for elite_prop in np.linspace(0.01, 0.3, 5):
                pop_size = int(pop_size)
                c.ES_POP_SIZE = pop_size
                c.ES_MUTATION_RATE = mutation_rate
                c.ES_ELITE_PROP = elite_prop
                c.ES_ITER = 30000 // pop_size

                mean, std = es.evaluate_strategy(n=5)
                time = es.get_mean_time()
                results.append((pop_size, mutation_rate, elite_prop, mean, std, time))

    results = sorted(results, key=lambda x: x[3], reverse=True)
    for result in results[:5]:
        print(f"Population size: {result[0]}, Mutation rate: {result[1]}, Elite proportion: {result[2]}, Mean: {result[3]}, Std: {result[4]}, Time: {result[5]}")
