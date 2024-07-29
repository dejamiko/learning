import logging
import time

import numpy as np

from al.mh import get_all_heuristics
from config import Config
from playground.environment import Environment
from utils import set_seed


class Solver:
    def __init__(self, config, heuristic_class):
        self.config = config
        self.heuristic_class = heuristic_class
        self._times_taken_on_strategy = []
        self.objects = []
        self.env = None
        self.heuristic = None

    def solve(self, n=5):
        counts = []
        for i in range(n):
            self._init_data(i)
            start = time.time()
            selected = self.heuristic.solve()
            end = time.time()
            count = self.env.evaluate_selection_transfer_based(selected)
            # early stopping for when the score is lower than the number of known objects
            if count < self.config.DEMONSTRATION_BUDGET:
                return -1, -1
            counts.append(count)
            self._times_taken_on_strategy.append(end - start)
        return np.mean(counts), np.std(counts)

    def get_mean_time(self):
        return np.mean(self._times_taken_on_strategy)

    def _init_data(self, i):
        set_seed(i)
        self.config.SEED = i
        self.env = Environment(self.config)
        self.objects = self.env.get_objects()
        self.heuristic = self.heuristic_class(self.config, self.env, [])


def evaluate_heuristic(solver_class, config, heuristic, n=100):
    print(f"Evaluating {heuristic.__name__} with config {config}")
    solver = solver_class(config, heuristic)
    mean, std = solver.solve(n)
    return mean, std, solver.get_mean_time()


def evaluate_all_heuristics(solver, config, n=100):
    results = []
    for h in get_all_heuristics():
        mean, std, total_time = evaluate_heuristic(solver, config, h, n)
        results.append((h.__name__, mean, std, total_time))
    return results


if __name__ == "__main__":
    c = Config()

    stopit_logger = logging.getLogger("stopit")
    stopit_logger.setLevel(logging.ERROR)

    results = evaluate_all_heuristics(Solver, c, n=1)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")
