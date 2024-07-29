from pprint import pprint

from al.mh.metaheuristic import MetaHeuristic
from config import Config
from playground.environment import Environment
from utils import set_seed, get_object_indices


class RandomSearchIter(MetaHeuristic):
    def __init__(self, c, environment, locked_subsolution, threshold=None):
        super().__init__(c, environment, locked_subsolution, threshold)
        self.best_selection = None

    def strategy(self):
        self.best_selection = self.get_random_selection()
        best_score = self.evaluate_selection(self.best_selection)

        while True:
            selected = self.get_random_selection()
            current_score = self.evaluate_selection(selected)
            if current_score > best_score:
                self.best_selection = selected
                best_score = current_score

    def get_best_solution(self):
        return self.best_selection


class RandomSearch(MetaHeuristic):
    def __init__(self, c, environment, locked_subsolution, threshold=None):
        super().__init__(c, environment, locked_subsolution, threshold)
        self.best_selection = None

    def strategy(self):
        self.best_selection = self.get_random_selection()

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    config = Config()
    set_seed(config.SEED)
    env = Environment(config)
    rs = RandomSearchIter(config, env, [])

    selected = rs.solve()
    print("Selected", get_object_indices(selected))
    config.USE_TRANSFER_EVALUATION = True
    print(rs.evaluate_selection_with_constraint_penalty(selected))
    pprint(env.get_reachable_object_indices(get_object_indices(selected)))

    config.USE_TRANSFER_EVALUATION = False
    rs = RandomSearch(config, env, [])

    selected = rs.solve()
    config.USE_TRANSFER_EVALUATION = True
    print(rs.evaluate_selection_with_constraint_penalty(selected))
