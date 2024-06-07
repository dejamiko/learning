import numpy as np

from active_learning import evaluate_selection, evaluate_strategy
from config import Config
from collections import deque


def get_neighbours(x, object_indices):
    neighbours = []
    not_selected = set(object_indices) - set(x)
    for i in x:
        for j in not_selected:
            neighbour = x.copy()
            # neighbour is an np array
            neighbour[neighbour == i] = j
            neighbours.append(neighbour)
    return neighbours


def tabu_search(objects, c, similarities):
    object_indices = np.arange(c.OBJ_NUM)
    selected = np.random.choice(object_indices, c.KNOWN_OBJECT_NUM, replace=False)
    L = c.TS_L
    s_best = selected
    best_candidate = selected

    tabu_list = deque(maxlen=L)
    tabu_list.append(set(selected))

    for k in range(c.TS_ITER):
        neighbours = get_neighbours(best_candidate, object_indices)
        best_candidate_fitness = np.NINF
        for n in neighbours:
            if any([set(n) == t for t in tabu_list]):
                continue
            fitness = evaluate_selection(objects, objects[n], similarities, c)
            if fitness > best_candidate_fitness:
                best_candidate = n
                best_candidate_fitness = fitness
        if best_candidate_fitness > np.NINF:
            s_best = best_candidate
        tabu_list.append(set(best_candidate))
    return objects[s_best]


if __name__ == "__main__":
    c = Config()

    # print(f"Tabu search selection for L={c.TS_L}")
    # mean, std = evaluate_strategy(tabu_search, c)
    # print(f"Mean: {mean}, std: {std}")

    for l in range(10000, 100000, 10000):
        c.TS_L = l
        print(f"Tabu search selection for L={c.TS_L}")
        mean, std = evaluate_strategy(tabu_search, c)
        print(f"Mean: {mean}, std: {std}")
