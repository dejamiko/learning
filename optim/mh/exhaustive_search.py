import time
from itertools import combinations

from config import Config
from optim.mh.metaheuristic import MetaHeuristic
from playground.environment import Environment
from utils import get_bin_representation, set_seed


class ExhaustiveSearch(MetaHeuristic):
    def __init__(self, c, environment, locked_subsolution, threshold=None):
        super().__init__(c, environment, locked_subsolution, threshold)
        self.best_selection = None

    def strategy(self):
        best_score = 0
        for selected in combinations(
            range(self.c.OBJ_NUM), self.c.DEMONSTRATION_BUDGET
        ):
            selected = get_bin_representation(list(selected), self.c.OBJ_NUM)
            score = self.evaluate_selection(selected)
            if score > best_score:
                best_score = score
                self.best_selection = selected

        return self.best_selection

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    config = Config()
    config.DEMONSTRATION_BUDGET = 10
    config.OBJ_NUM = 20
    config.MH_BUDGET = 2000000
    set_seed(config.SEED)
    env = Environment(config)
    es = ExhaustiveSearch(config, env, [])

    start = time.time()
    selected = es.strategy()
    print(time.time() - start)
    print(es.evaluate_selection(selected))
