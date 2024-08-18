from collections import deque

from config import Config
from optim.mh.metaheuristic import MetaHeuristic
from optim.utils import NeighbourGenerator
from playground.environment import Environment
from tm_utils import get_object_indices


class TabuList:
    def __init__(self, c):
        self.L = c.TS_L
        self.tabu_set = set()
        self.tabu_list = deque()
        self.c = c

    def add(self, neighbour):
        if len(self.tabu_list) == self.L:
            neighbour_to_remove = self.tabu_list.popleft()
            self.tabu_set.remove(neighbour_to_remove)
        neighbour_z = tuple(neighbour)
        self.tabu_list.append(neighbour_z)
        self.tabu_set.add(neighbour_z)

    def is_tabu(self, neighbour):
        return tuple(neighbour) in self.tabu_set


class TabuSearch(MetaHeuristic):
    def __init__(self, c, similarity_dict, locked_subsolution):
        super().__init__(c, similarity_dict, locked_subsolution)
        self.tabu_list = TabuList(self.c)
        self.best_selection = None

    def strategy(self):
        initial = self.get_random_selection()
        self.best_selection = initial
        g_best = self.evaluate_selection(initial)

        candidate = initial
        candidate_swap = None

        while self.count < self.c.MH_BUDGET:
            neighbour_gen = NeighbourGenerator(
                candidate, self.locked_subsolution, self._rng
            )
            best_neighbour_fitness = -float("inf")
            for neighbour, swap in neighbour_gen:
                if self.count >= self.c.MH_BUDGET:
                    break
                neighbour_fitness = self.evaluate_selection(neighbour)
                if not self.tabu_list.is_tabu(swap):
                    if neighbour_fitness > best_neighbour_fitness:
                        candidate = neighbour
                        best_neighbour_fitness = neighbour_fitness
                        candidate_swap = swap
                elif neighbour_fitness > g_best:  # Aspiration criteria
                    candidate = neighbour
                    best_neighbour_fitness = neighbour_fitness
                    candidate_swap = swap
            if best_neighbour_fitness == -float("inf"):  # No valid move found
                break
            if best_neighbour_fitness > g_best:
                g_best = best_neighbour_fitness
                self.best_selection = candidate

            self.tabu_list.add(candidate_swap)
        return self.best_selection

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    for t_size in [1, 10, 100, 1000, 10000]:
        config = Config()
        config.TS_L = t_size
        env = Environment(config)
        ts = TabuSearch(config, env, [])

        selected = ts.strategy()
        print("+" * 25)
        print(ts.evaluate_selection(selected))
        print(get_object_indices(selected))
        print("+" * 25)
