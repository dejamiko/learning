from collections import deque

from al.mh.metaheuristic import MetaHeuristic
from al.utils import NeighbourGenerator
from config import Config
from playground.environment import Environment
from utils import set_seed


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
    def __init__(self, c, similarity_dict, locked_subsolution, threshold=None):
        super().__init__(c, similarity_dict, locked_subsolution, threshold)
        self.tabu_list = TabuList(self.c)
        self.best_selection = None

    def strategy(self):
        selected = self.get_random_initial_selection()
        g_best = self.evaluate_selection(selected)
        self.best_selection = selected
        g_s = g_best

        while self.count < self.c.MH_BUDGET:
            neighbour_gen = NeighbourGenerator(selected, self.locked_subsolution)
            current = None
            g_n = 0
            for neighbour in neighbour_gen:
                if self.count >= self.c.MH_BUDGET:
                    break
                current = neighbour
                g_n = self.evaluate_selection(neighbour)
                delta = g_n - g_s
                if (
                    delta > -self.c.TS_GAMMA and not self.tabu_list.is_tabu(neighbour)
                ) or g_n > g_best:
                    break
            selected = current
            self.tabu_list.add(selected)
            if g_n > g_best:
                g_best = g_n
                self.best_selection = selected
        return self.best_selection

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    config = Config()
    set_seed(config.SEED)
    env = Environment(config)
    ts = TabuSearch(config, env, [])

    selected = ts.strategy()
    print(ts.evaluate_selection(selected))
