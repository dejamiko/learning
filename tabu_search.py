import time
from collections import deque

import numpy as np

from active_learning import evaluate_selection, evaluate_strategy
from config import Config


# def get_neighbours(x, object_indices):
#     neighbours = []
#     not_selected = set(object_indices) - set(x)
#     for i in x:
#         for j in not_selected:
#             neighbour = x.copy()
#             # neighbour is an np array
#             neighbour[neighbour == i] = j
#             neighbours.append(neighbour)
#     return neighbours


class NeighbourGenerator:
    def __init__(self, object_indices, selected):
        self.selected = selected
        self.not_selected = list(set(object_indices) - set(selected))

    def __iter__(self):
        self.selected = np.random.permutation(self.selected)
        self.not_selected = np.random.permutation(self.not_selected)
        for i in self.selected:
            for j in self.not_selected:
                neighbour = self.selected.copy()
                neighbour[neighbour == i] = j
                yield neighbour


class TabuList:
    def __init__(self, length, c):
        self.L = length
        self.tabu_set = set()
        self.tabu_list = deque()
        self.c = c

    def add(self, neighbour):
        if len(self.tabu_list) == self.L:
            neighbour_to_remove = self.tabu_list.popleft()
            self.tabu_set.remove(neighbour_to_remove)
        # create a one-hot encoding of the neighbour
        neighbour_z = np.zeros(self.c.OBJ_NUM)
        neighbour_z[neighbour] = 1
        neighbour_z = tuple(neighbour_z)
        self.tabu_list.append(neighbour_z)
        self.tabu_set.add(neighbour_z)

    def is_tabu(self, neighbour):
        neighbour_z = np.zeros(self.c.OBJ_NUM)
        neighbour_z[neighbour] = 1
        neighbour_z = tuple(neighbour_z)
        return neighbour_z in self.tabu_set


def get_neighbour(selected, object_indices):
    i = np.random.choice(selected)
    not_selected = set(object_indices) - set(selected)
    j = np.random.choice(list(not_selected))
    neighbour = selected.copy()
    neighbour[neighbour == i] = j
    return neighbour, i, j


def tabu_search(objects, c, similarities):
    object_indices = np.arange(c.OBJ_NUM)
    selected = np.random.choice(object_indices, c.KNOWN_OBJECT_NUM, replace=False)
    L = c.TS_L
    g_best = evaluate_selection(selected, similarities, c)
    s_best = selected
    g_s = g_best
    tabu_list = TabuList(L, c)

    for k in range(c.TS_ITER):
        neighbour_gen = NeighbourGenerator(object_indices, selected)
        for neighbour in neighbour_gen:
            g_n = evaluate_selection(neighbour, similarities, c)
            delta = g_n - g_s
            if (delta > - c.TS_GAMMA and not tabu_list.is_tabu(neighbour)) or g_n > g_best:
                break
        selected = neighbour
        tabu_list.add(neighbour)
        if g_n > g_best:
            g_best = g_n
            s_best = selected
    return s_best


if __name__ == "__main__":
    c = Config()

    # print(f"Tabu search selection for L={c.TS_L}")
    # mean, std = evaluate_strategy(tabu_search, c)
    # print(f"Mean: {mean}, std: {std}")

    for l in [10, 100, 1000, 10000, 100000]:
        start_time = time.time()
        c.TS_L = l
        print(f"Tabu search selection for L={c.TS_L}")
        mean, std = evaluate_strategy(tabu_search, c, n=5)
        print(f"Mean: {mean}, std: {std}")
        print(f"Time taken: {time.time() - start_time}")
