import time
from abc import ABC, abstractmethod

import numpy as np

from optim.mh import get_all_heuristics
from playground.environment import Environment
from tm_utils import get_rng


class Solver(ABC):
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
            count = self.solve_one()
            counts.append(count)
            self._times_taken_on_strategy.append(time.time() - start)
        return np.mean(counts), np.std(counts)

    @abstractmethod
    def solve_one(self):
        pass

    def get_mean_time(self):
        return np.mean(self._times_taken_on_strategy)

    def _init_data(self, i):
        self._rng = get_rng(i)
        self.config.SEED = i
        self.env = Environment(self.config)
        self.objects = self.env.get_objects()
        self.heuristic = self.heuristic_class(self.config, self.env, [])


def evaluate_heuristic(solver_class, config, heuristic, n=1):
    solver = solver_class(config, heuristic)
    mean, std = solver.solve(n)
    return mean, std, solver.get_mean_time()


def evaluate_all_heuristics(solver, config, n=1):
    results = []
    for h in get_all_heuristics():
        mean, std, total_time = evaluate_heuristic(solver, config, h, n)
        results.append((h.__name__, mean, std, total_time))
    return results


def evaluate_provided_heuristics(solver, config, n=1, heuristics=()):
    results = []
    for h in heuristics:
        mean, std, total_time = evaluate_heuristic(solver, config, h, n)
        results.append((h.__name__, mean, std, total_time))
    return results
