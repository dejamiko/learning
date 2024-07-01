import logging
import time

import numpy as np

from al.mh import get_all_heuristics
from al.utils import set_seed
from config import Config
from playground.environment import Environment


class Solver:
    def __init__(self, config, heuristic_class):
        self.config = config
        self.heuristic_class = heuristic_class
        self._times_taken_on_strategy = []
        self.objects = []
        self.environment = None
        self.heuristic = None

    def solve(self, n=5):
        counts = []
        for i in range(n):
            self._init_data(i)
            start = time.time()
            selected = self.heuristic.solve()
            end = time.time()
            count = self.evaluate(selected)
            # early stopping for when the score is lower than the number of known objects
            if count < self.config.KNOWN_OBJECT_NUM:
                return -1, -1
            counts.append(count)
            self._times_taken_on_strategy.append(end - start)
        return np.mean(counts), np.std(counts)

    def _init_data(self, i):
        set_seed(i)
        self.config.SEED = i
        self.environment = Environment(self.config)
        self.objects = self.environment.get_objects()
        self.heuristic = self.heuristic_class(self.config, self.environment, [])

    def evaluate(self, selected):
        if self.config.USE_TRANSFER_EVALUATION:
            count = self.environment.evaluate_selection_transfer_based(selected)
        else:
            count = self.environment.evaluate_selection_similarity_based(selected)
        return count

    def get_mean_time(self):
        return np.mean(self._times_taken_on_strategy)

    def get_total_time(self):
        return np.sum(self._times_taken_on_strategy)


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
    c.TASK_TYPES = ["sample task"]

    stopit_logger = logging.getLogger("stopit")
    stopit_logger.setLevel(logging.ERROR)

    results = evaluate_all_heuristics(Solver, c, n=100)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")

    c.USE_TRANSFER_EVALUATION = True
    results = evaluate_all_heuristics(Solver, c, n=100)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")
