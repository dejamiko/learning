import numpy as np

from active_learning import evaluate_selection, evaluate_strategy
from config import Config
from tabu_search import NeighbourGenerator


def randomised_hill_climbing(objects, c, similarities):
    object_indices = np.arange(c.OBJ_NUM)
    best_selection = None
    best_score = 0
    for i in range(c.RHC_ITER):
        selected = np.random.choice(object_indices, c.KNOWN_OBJECT_NUM, replace=False)
        curr_score = evaluate_selection(selected, similarities, c)

        while True:
            next_selection = None
            next_score = 0
            for neighbour in NeighbourGenerator(object_indices, selected):
                score = evaluate_selection(neighbour, similarities, c)
                if score > next_score:
                    next_selection = neighbour
                    next_score = score
            if next_score <= curr_score:
                if curr_score > best_score:
                    best_selection = selected
                    best_score = curr_score
                break
            selected = next_selection
            curr_score = next_score

    return best_selection


if __name__ == "__main__":
    c = Config()

    print(f"Randomised hill climbing selection for {c.RHC_ITER} iterations")
    mean, std = evaluate_strategy(randomised_hill_climbing, c, n=5)
    print(f"Mean: {mean}, std: {std}")
