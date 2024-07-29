from typing import Union, List, Tuple

import mealpy
import numpy as np
from mealpy import Problem

from al.mh.metaheuristic import MetaHeuristic
from config import Config
from playground.environment import Environment
from utils import set_seed


class ObjectProblem(Problem):
    def __init__(self, bounds=None, minmax="max", heuristic=None, **kwargs):
        self.heuristic = heuristic
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x: np.ndarray) -> Union[List, Tuple, np.ndarray, int, float]:
        x_decoded = self.decode_solution(x)
        x = x_decoded["object_selection"]

        return self.heuristic.evaluate_selection_with_constraint_penalty(x)


class MealpyHeuristic(MetaHeuristic):
    def __init__(self, c, similarity_dict, locked_subsolution, threshold=None):
        super().__init__(c, similarity_dict, locked_subsolution, threshold)
        self.best_selection = None

    def strategy(self):
        lb = [0] * self.c.OBJ_NUM
        for i in self.locked_subsolution:
            lb[i] = 1
        ub = [1] * self.c.OBJ_NUM
        bounds = mealpy.IntegerVar(lb=lb, ub=ub, name="object_selection")
        problem = ObjectProblem(
            bounds=bounds, heuristic=self, minmax="max", log_to=None, seed=self.c.SEED
        )
        self.optimizer = mealpy.get_optimizer_by_name(self.c.MP_OPTIMISER_NAME)()
        termination = mealpy.Termination(max_fe=self.c.MH_BUDGET)
        self.optimizer.solve(problem, termination=termination)
        final = self.optimizer.problem.decode_solution(self.optimizer.g_best.solution)[
            "object_selection"
        ]
        if np.sum(final) != self.c.DEMONSTRATION_BUDGET:
            self.best_selection = self.get_random_selection()
        else:
            self.best_selection = final
        return self.best_selection

    def get_best_solution(self):
        if self.best_selection is not None:
            return self.best_selection
        final = self.optimizer.problem.decode_solution(self.optimizer.g_best.solution)[
            "object_selection"
        ]
        if np.sum(final) != self.c.DEMONSTRATION_BUDGET:
            self.best_selection = self.get_random_selection()
        else:
            self.best_selection = final
        return self.best_selection


if __name__ == "__main__":
    config = Config()
    set_seed(config.SEED)
    env = Environment(config)
    mealpyh = MealpyHeuristic(config, env, [])

    selected = mealpyh.strategy()
    print(mealpyh.evaluate_selection_with_constraint_penalty(selected))

    # Optimizer: DevSPBO, Mean: 27.92, Std: 2.7501999927278016
    # Optimizer: OriginalWarSO, Mean: 26.83, Std: 2.1612727731593715
    # Optimizer: OriginalWaOA, Mean: 25.64, Std: 2.1495115724275595
    # Optimizer: DevSCA, Mean: 25.455, Std: 2.2153949986401975
    # Optimizer: OriginalServalOA, Mean: 25.31, Std: 2.4827202822710417
    # Optimizer: OriginalNGO, Mean: 24.655, Std: 2.162400286718442
    # Optimizer: OriginalCoatiOA, Mean: 23.29, Std: 2.1762123058194485
    # Optimizer: AdaptiveBA, Mean: 22.355, Std: 2.3935277311951078
    # Optimizer: ImprovedSFO, Mean: 22.125, Std: 2.2758240265890506
    # Optimizer: OriginalGTO, Mean: 21.605, Std: 2.6036464813795286
    # Optimizer: DevBRO, Mean: 19.955, Std: 2.419292251878636
    # Optimizer: OriginalFFA, Mean: 19.775, Std: 2.331174596635782
