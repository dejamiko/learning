from typing import Union, List, Tuple

import mealpy
import numpy as np
from mealpy import Problem

from al.utils import MetaHeuristic
from config import Config


class ObjectProblem(Problem):
    def __init__(self, bounds=None, minmax="max", heuristic=None, **kwargs):
        self.heuristic = heuristic
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x: np.ndarray) -> Union[List, Tuple, np.ndarray, int, float]:
        x_decoded = self.decode_solution(x)
        x = x_decoded["object_selection"]

        if np.sum(x) > self.heuristic.c.KNOWN_OBJECT_NUM:
            return -np.sum(x)
        return self.heuristic.evaluate_selection(x)


class MealpyHeuristic(MetaHeuristic):
    def __init__(self, c, threshold=None, optimizer_name="OriginalWOA"):
        super().__init__(c, threshold)
        self.optimizer_name = optimizer_name

    def f(self, x):
        if np.sum(x) > self.c.KNOWN_OBJECT_NUM:
            return -np.sum(x)
        return self.evaluate_selection(x)

    def strategy(self):
        bounds = mealpy.IntegerVar(lb=[0, ] * self.c.OBJ_NUM, ub=[1, ] * self.c.OBJ_NUM, name="object_selection")
        problem = ObjectProblem(bounds=bounds, heuristic=self, minmax="max", log_to=None)
        optimizer = mealpy.get_optimizer_by_name(self.optimizer_name)()
        termination = mealpy.Termination(max_fe=self.c.MH_BUDGET)
        optimizer.solve(problem, termination=termination)
        final = optimizer.problem.decode_solution(optimizer.g_best.solution)["object_selection"]
        if np.sum(final) > self.c.KNOWN_OBJECT_NUM:
            return np.zeros(self.c.OBJ_NUM)
        return final


if __name__ == "__main__":
    config = Config()

    results = []

    for optimizer_name in mealpy.get_all_optimizers().keys():
        try:
            mealpy_heuristic = MealpyHeuristic(config, optimizer_name=optimizer_name)
            mean, std = mealpy_heuristic.evaluate_strategy(n=200)
            if mean > 0:
                results.append((optimizer_name, mean, std))
        except Exception as e:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    for result in results:
        print(f"Optimizer: {result[0]}, Mean: {result[1]}, Std: {result[2]}")
