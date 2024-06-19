import time

import numpy as np

from al.mh import get_all_heuristics
from al.utils import get_object_indices, set_seed
from config import Config


class Solver:
    def __init__(self, config, heuristic):
        self.config = config
        self.heuristic = heuristic
        self._times_taken_on_strategy = []
        self.objects = []
        self.environment = None

    def get_real_evaluation(self, selected):
        selected = get_object_indices(selected)
        count = 0
        selected = self.objects[selected]
        for o in self.objects:
            for s in selected:
                if self.environment.try_transfer(o, s):
                    count += 1
                    break
        return count

    def get_predicted_evaluation(self, selected):
        return self.heuristic.evaluate_selection(selected)

    def solve(self, n=5):
        counts = []
        for i in range(n):
            set_seed(i)
            self.config.SEED = i
            self.heuristic.initialise_data()
            self.objects = self.heuristic.get_objects()
            self.environment = self.heuristic.get_environment()
            start = time.time()
            selected = self.heuristic.strategy()
            end = time.time()
            count = self.evaluate(selected)
            # early stopping for when the score is lower than the number of known objects
            if count < self.config.KNOWN_OBJECT_NUM:
                return -1, -1
            counts.append(count)
            self._times_taken_on_strategy.append(end - start)
        return np.mean(counts), np.std(counts)

    def evaluate(self, selected):
        if self.config.USE_ACTUAL_EVALUATION:
            count = self.get_real_evaluation(selected)
        else:
            count = self.get_predicted_evaluation(selected)
        return count

    def get_mean_time(self):
        return np.mean(self._times_taken_on_strategy)

    def get_total_time(self):
        return np.sum(self._times_taken_on_strategy)


def evaluate_heuristic(solver_class, config, heuristic, n=100):
    solver = solver_class(config, heuristic)
    mean, std = solver.solve(n)
    return mean, std, solver.get_mean_time()


def evaluate_all_heuristics(solver, config, n=100):
    results = []
    for h in get_all_heuristics():
        heuristic = h(config)
        mean, std, total_time = evaluate_heuristic(solver, config, heuristic, n)
        results.append((h.__name__, mean, std, total_time))
    return results


if __name__ == "__main__":
    c = Config()
    c.TASK_TYPES = ["sample task"]

    results = evaluate_all_heuristics(Solver, c, n=500)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")

    c.USE_ACTUAL_EVALUATION = True
    results = evaluate_all_heuristics(Solver, c, n=500)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")
