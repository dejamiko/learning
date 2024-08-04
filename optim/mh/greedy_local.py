import numpy as np

from config import Config
from optim.mh.metaheuristic import MetaHeuristic
from playground.environment import Environment
from tm_utils import Task


class GreedyLocalSearch(MetaHeuristic):
    def __init__(self, c, environment, locked_subsolution):
        super().__init__(c, environment, locked_subsolution)
        self.best_selection = None

    def strategy(self):
        selected = np.zeros(self.c.OBJ_NUM)
        objects = self.environment.get_objects()
        while np.sum(selected) < self.c.DEMONSTRATION_BUDGET:
            # find an unused object that transfers to the largest number of objects
            max_reachable_count = -1
            best_object_index = -1
            for o in objects:
                if selected[o.index] == 1:
                    continue
                _, sim = self.environment.similarity_dict[o.index]
                ind = np.searchsorted(
                    sim, self.c.SIMILARITY_THRESHOLDS[Task.get_ind(o.task)]
                )
                if ind > max_reachable_count:
                    max_reachable_count = ind
                    best_object_index = o.index
            selected[best_object_index] = 1
        self.best_selection = selected
        return self.best_selection

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    config = Config()
    env = Environment(config)
    gls = GreedyLocalSearch(config, env, [])

    selected = gls.strategy()
    print(gls.evaluate_selection_with_constraint_penalty(selected))
