import numpy as np

from active_learning import evaluate_selection, evaluate_strategy
from config import Config


def get_cost(objects, selected, similarities, c):
    return len(objects) - evaluate_selection(selected, similarities, c)


def get_neighbour(objects, selected):
    pool = set(objects) - set(selected)
    addition = np.random.choice(list(pool), 1, replace=False)[0]
    removal = np.random.choice(selected, 1, replace=False)[0]
    selected = list(selected)
    selected.remove(removal)
    selected.append(addition)
    return selected


def simulated_annealing(objects, c, similarities):
    object_indices = np.arange(c.OBJ_NUM)
    selected = np.random.choice(object_indices, c.KNOWN_OBJECT_NUM, replace=False)
    best_selection = selected
    best_cost = get_cost(objects, selected, similarities, c)
    T = c.SA_T
    alpha = c.SA_ALPHA
    curr = best_selection
    for k in range(c.SA_ITER):
        new_selection = get_neighbour(object_indices, curr)
        new_cost = get_cost(objects, new_selection, similarities, c)

        diff = best_cost - new_cost

        if diff >= 0:
            best_selection = new_selection
        elif np.random.uniform() <= np.exp(diff / T):
            best_selection = new_selection

        T = T * alpha
        T = max(T, c.SA_T_MIN)
        curr = new_selection
        if new_cost < best_cost:
            best_cost = new_cost
            best_selection = new_selection

    return best_selection


if __name__ == "__main__":
    c = Config()

    print(f"Simulated annealing selection for alpha={c.SA_ALPHA}, T={c.SA_T}")
    mean, std = evaluate_strategy(simulated_annealing, c, n=5)
    print(f"Mean: {mean}, std: {std}")

    for t in range(1, 100, 10):
        for alpha in np.arange(0.8, 1, 0.01):
            c.SA_T = t
            c.SA_ALPHA = alpha

            print(f"Simulated annealing selection for alpha={alpha}, T={t}")
            mean, std = evaluate_strategy(simulated_annealing, c, n=5)
            print(f"Mean: {mean}, std: {std}")
