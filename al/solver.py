"""
This file contains the Solver class for solving the problem based on the visual similarity of the objects but
the final evaluation is based on the latent similarity of the objects.
"""

import time

import numpy as np

from al.evolutionary_strategy import EvolutionaryStrategy
from al.randomised_hill_climbing import RandomisedHillClimbing
from al.sa import SimulatedAnnealing
from al.tabu_search import TabuSearch
from al.utils import get_object_indices, get_bin_representation
from config import Config
from playground.environment import Environment
from playground.object import TrajectoryObject


class Solver:
    def __init__(self, config, heuristic):
        self.config = config
        self.environment = Environment(config)
        self.environment.generate_objects_ail(TrajectoryObject)
        self.heuristic = heuristic(config)
        self.heuristic.generate_similarity_dict(self.environment)
        self.objects = self.environment.get_objects()

    def solve(self):
        selected = self.heuristic.strategy()
        return get_object_indices(selected)

    def evaluate(self, selected):
        count = 0
        selected = self.objects[selected]
        for o in self.objects:
            for s in selected:
                if (
                    s.task_type == o.task_type
                    and self.environment.get_latent_similarity(o, s)
                    > self.config.SIMILARITY_THRESHOLD
                ):
                    count += 1
                    break
        return count

    def get_predicted_evaluation(self, selected):
        selected = get_bin_representation(selected, self.config.OBJ_NUM)
        return self.heuristic.evaluate_selection(selected)


if __name__ == "__main__":
    c = Config()
    c.TASK_TYPES = ["sample task"]
    c.OBJ_NUM = 100
    c.LATENT_DIM = 10
    c.MH_BUDGET = 20000

    seeds = np.arange(10)
    heuristics = [
        SimulatedAnnealing,
        RandomisedHillClimbing,
        TabuSearch,
        EvolutionaryStrategy,
    ]
    for h in heuristics:
        solved, times = 0, 0
        start = time.time()
        for seed in seeds:
            np.random.seed(seed)
            solver = Solver(c, h)
            selected = solver.solve()
            solved += solver.evaluate(selected)
            times += solver.heuristic.count
        end = time.time()
        print(
            f"{h.__name__} solved {solved / len(seeds)} with average {times / len(seeds)} evaluations, taking {end - start} seconds."
        )
