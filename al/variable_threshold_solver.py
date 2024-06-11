"""
This file contains the Solver class for solving the problem based on the visual similarity of the objects but
the final evaluation is based on the latent similarity of the objects. The threshold for the similarity is not known
and is estimated based on the interactions of the objects.
"""

import numpy as np

from al.evolutionary_strategy import EvolutionaryStrategy
from al.randomised_hill_climbing import RandomisedHillClimbing
from al.sa import SimulatedAnnealing
from al.tabu_search import TabuSearch
from al.utils import get_bin_representation, get_object_indices
from config import Config
from playground.environment import Environment
from playground.object import TrajectoryObject


class Solver:
    def __init__(self, config, heuristic, lower_bound=0.0, upper_bound=1.0):
        self.config = config
        self.environment = Environment(config)
        self.environment.generate_objects_ail(TrajectoryObject)
        self.threshold_lower_bound = lower_bound
        self.threshold_upper_bound = upper_bound
        self.heuristic = heuristic(
            config, (self.threshold_lower_bound + self.threshold_upper_bound) / 2
        )
        self.heuristic.generate_similarity_dict(self.environment)
        self.objects = self.environment.get_objects()

    def solve(self):
        selected = []
        while len(selected) < self.config.KNOWN_OBJECT_NUM:
            heuristic_selected = self.heuristic.strategy()
            obj_to_try = self.select_object_to_try(heuristic_selected)
            assert heuristic_selected[obj_to_try] == 1
            selected.append(obj_to_try)
            self.heuristic.lock_object(obj_to_try)
            # update the lower and upper bounds based on the interactions of the selected object
            self.update_bounds(obj_to_try)
        return selected

    def evaluate(self, selected):
        count = 0
        selected = self.objects[selected]
        for o in self.objects:
            for s in selected:
                if self.environment.try_transfer(o, s):
                    count += 1
                    break
        return count

    def get_predicted_evaluation(self, selected):
        selected = get_bin_representation(selected, self.config.OBJ_NUM)
        return self.heuristic.evaluate_selection(selected)

    def select_object_to_try(self, selected):
        selected = get_object_indices(selected)
        # go through the objects and select the one that maximizes the expected information gain
        if self.config.THRESH_ESTIMATION_STRAT == "density":
            return self._density_selection(selected)
        elif self.config.THRESH_ESTIMATION_STRAT == "random":
            to_choose = set(selected) - set(self.heuristic.locked_subsolution)
            return np.random.choice(list(to_choose))
        elif self.config.THRESH_ESTIMATION_STRAT == "intervals":
            return self._interval_selection(selected)
        elif self.config.THRESH_ESTIMATION_STRAT == "greedy":
            return self._greedy_selection(selected)
        else:
            raise ValueError(
                f"Unknown threshold estimation strategy: {self.config.THRESH_ESTIMATION_STRAT}"
            )

    def _density_selection(self, selected):
        # this can be estimated by selecting one that has the most objects that have a similarity between the bounds
        to_search = list(set(selected) - set(self.heuristic.locked_subsolution))
        best_object_index = np.random.choice(to_search)
        best_score = 0
        for s in to_search:
            score = 0
            for o in self.objects:
                if (
                    self.threshold_lower_bound
                    < self.environment.get_visual_similarity(self.objects[s], o)
                    < self.threshold_upper_bound
                ) and self.objects[s].task_type == o.task_type:
                    score += 1
            if score > best_score:
                best_score = score
                best_object_index = s
        return best_object_index

    def _interval_selection(self, selected):
        to_search = list(set(selected) - set(self.heuristic.locked_subsolution))
        best_object_index = np.random.choice(to_search)
        best_score = np.Inf
        for s in to_search:
            sim_objects_between_bounds = []
            for o in self.objects:
                if (
                    self.threshold_lower_bound
                    < self.environment.get_visual_similarity(self.objects[s], o)
                    < self.threshold_upper_bound
                ) and self.objects[s].task_type == o.task_type:
                    sim_objects_between_bounds.append(
                        self.environment.get_latent_similarity(self.objects[s], o)
                    )
            sim_objects_between_bounds.sort()
            for i in range(len(sim_objects_between_bounds) - 1):
                score = (
                    sim_objects_between_bounds[i + 1] - sim_objects_between_bounds[i]
                )
                if score < best_score:
                    best_score = score
                    best_object_index = s
        return best_object_index

    def _greedy_selection(self, selected):
        # find the selected object that has the most objects that are below the estimated threshold
        to_search = list(set(selected) - set(self.heuristic.locked_subsolution))
        best_object_index = np.random.choice(to_search)
        best_score = np.NINF
        for s in to_search:
            score = 0
            for o in self.objects:
                if (
                    self.environment.get_visual_similarity(self.objects[s], o)
                    < (self.threshold_lower_bound + self.threshold_upper_bound) / 2
                ):
                    score += 1
            if score > best_score:
                best_score = score
                best_object_index = s
        return best_object_index

    def update_bounds(self, obj_ind):
        successes = []
        failures = []
        obj = self.objects[obj_ind]
        for o in self.objects:
            if self.environment.try_transfer(obj, o):
                successes.append(o)
            else:
                failures.append(o)
        for s in successes:
            sim = self.environment.get_latent_similarity(obj, s)
            self.threshold_upper_bound = min(self.threshold_upper_bound, sim)
        for f in failures:
            sim = self.environment.get_latent_similarity(obj, f)
            self.threshold_lower_bound = max(self.threshold_lower_bound, sim)
        if self.config.VERBOSITY > 0:
            print(
                f"Lower bound: {self.threshold_lower_bound}, upper bound: {self.threshold_upper_bound}, real threshold: {self.config.SIMILARITY_THRESHOLD}"
            )
        self.heuristic.update_threshold_estimate(
            (self.threshold_lower_bound + self.threshold_upper_bound) / 2
        )


def _run_experiment(method, heuristic, threshold=False):
    np.random.seed(s)
    c = Config()
    c.THRESH_ESTIMATION_STRAT = method
    c.TASK_TYPES = ["sample task"]
    c.OBJ_NUM = 100
    c.LATENT_DIM = 10
    c.VERBOSITY = 0
    c.SIMILARITY_THRESHOLD = np.random.uniform(0.5, 0.9)
    if threshold:
        solver = Solver(c, heuristic, c.SIMILARITY_THRESHOLD, c.SIMILARITY_THRESHOLD)
    else:
        solver = Solver(c, heuristic)
    selected = solver.solve()
    return solver.evaluate(selected)


if __name__ == "__main__":
    for heuristic in [
        SimulatedAnnealing,
        TabuSearch,
        RandomisedHillClimbing,
        EvolutionaryStrategy,
    ]:
        print(f"Running experiments for heuristic: {heuristic.__name__}")
        avg_performance = 0
        for method in ["density", "random", "intervals", "greedy"]:
            counts = []
            for s in range(150):
                count = _run_experiment(method, heuristic)
                count2 = _run_experiment(method, heuristic, True)

                counts.append((count, count2))

            # find the average gain in performance by knowing the threshold
            total_without = 0
            total_using_known = 0
            for c1, c2 in counts:
                total_without += c1
                total_using_known += c2
            total_without /= len(counts)
            total_using_known /= len(counts)
            print(
                f"Method: {method} gave an average performance of {total_without} without knowing the threshold and "
                f"{total_using_known} with knowing the threshold"
            )
            print(
                f"Average gain in performance by knowing the threshold: {total_using_known - total_without}"
            )
            # also report this as a percentage of the number of objects successfully transferred
            print(
                "Average gain in performance by knowing the threshold as a percentage of the total: "
                f"{(total_using_known - total_without) / total_using_known * 100}%"
            )
            avg_performance += total_without
        avg_performance /= 4
        print(
            f"Average performance for heuristic {heuristic.__name__}: {avg_performance}"
        )
