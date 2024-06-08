from collections import deque

from config import Config
from utils import NeighbourGenerator, MetaHeuristic


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
    def __init__(self, c):
        super().__init__(c)
        self.tabu_list = TabuList(self.c)

    def strategy(self):
        selected = self._get_initial_selection()
        g_best = self.evaluate_selection(selected)
        self.best_selection = selected
        g_s = g_best

        for k in range(self.c.TS_ITER):
            neighbour_gen = NeighbourGenerator(selected)
            for neighbour in neighbour_gen:
                g_n = self.evaluate_selection(neighbour)
                delta = g_n - g_s
                if (delta > - self.c.TS_GAMMA and not self.tabu_list.is_tabu(neighbour)) or g_n > g_best:
                    break
            selected = neighbour
            self.tabu_list.add(neighbour)
            if g_n > g_best:
                g_best = g_n
                self.best_selection = selected
        return self.best_selection


if __name__ == "__main__":
    c = Config()

    ts = TabuSearch(c)

    for tabu_list_size in [10, 100, 1000, 10000, 100000]:
        c.TS_L = tabu_list_size
        print(f"Tabu search selection for L={c.TS_L}")
        mean, std = ts.evaluate_strategy()
        print(f"Mean: {mean}, std: {std}, time taken: {ts.get_mean_time()}")
