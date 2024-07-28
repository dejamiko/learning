import time
from itertools import combinations

from al.mh.metaheuristic import MetaHeuristic
from config import Config
from playground.environment import Environment
from utils import get_bin_representation, set_seed


class ExhaustiveSearch(MetaHeuristic):
    def __init__(self, c, environment, locked_subsolution, threshold=None):
        super().__init__(c, environment, locked_subsolution, threshold)
        self.best_selection = None

    def strategy(self):
        # if comb(self.c.OBJ_NUM, self.c.KNOWN_OBJECT_NUM) > self.c.MH_BUDGET:
        #     raise ValueError(f"Too many combinations to evaluate, "
        #                      f"{comb(self.c.OBJ_NUM, self.c.KNOWN_OBJECT_NUM)} > {self.c.MH_BUDGET}")

        best_score = 0
        for selected in combinations(range(self.c.OBJ_NUM), self.c.KNOWN_OBJECT_NUM):
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
    config.KNOWN_OBJECT_NUM = 10
    config.OBJ_NUM = 20
    config.MH_BUDGET = 2000000
    set_seed(config.SEED)
    env = Environment(config)
    es = ExhaustiveSearch(config, env, [])

    start = time.time()
    selected = es.strategy()
    print(time.time() - start)
    print(es.evaluate_selection(selected))
