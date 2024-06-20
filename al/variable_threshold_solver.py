import time

import numpy as np

from al.mh import get_all_heuristics
from al.solver import Solver, evaluate_all_heuristics
from al.utils import get_bin_representation, get_object_indices, set_seed
from config import Config


class VariableThresholdSolver(Solver):
    def __init__(self, config, heuristic_class):
        super().__init__(config, heuristic_class)
        self.threshold_lower_bound = 0.0
        self.threshold_upper_bound = 1.0

    def solve(self, n=5):
        counts = []
        for i in range(n):
            self._init_data(i)
            start = time.time()
            count = self.solve_one()
            counts.append(count)
            self._times_taken_on_strategy.append(time.time() - start)
        return np.mean(counts), np.std(counts)

    def _init_data(self, i):
        set_seed(i)
        self.reset_bounds()
        self.config.SEED = i
        self.config.SIMILARITY_THRESHOLD = np.random.uniform(0.5, 0.9)
        self.heuristic = self.heuristic_class(
            self.config, (self.threshold_lower_bound + self.threshold_upper_bound) / 2
        )
        self.heuristic.initialise_data()
        self.objects = self.heuristic.get_objects()
        self.environment = self.heuristic.get_environment()

    def solve_one(self):
        selected = []
        while len(selected) < self.config.KNOWN_OBJECT_NUM:
            self.heuristic.count = 0
            heuristic_selected = self.heuristic.strategy()
            obj_to_try = self.select_object_to_try(heuristic_selected)
            assert heuristic_selected[obj_to_try] == 1
            selected.append(obj_to_try)
            self.heuristic.lock_object(obj_to_try)
            # update the lower and upper bounds based on the interactions of the selected object
            self.update_bounds(obj_to_try)
        count = self.evaluate(selected)
        return count

    def reset_bounds(self):
        if self.config.USE_REAL_THRESHOLD:
            self.threshold_lower_bound = self.config.SIMILARITY_THRESHOLD
            self.threshold_upper_bound = self.config.SIMILARITY_THRESHOLD
        else:
            self.threshold_lower_bound = 0.0
            self.threshold_upper_bound = 1.0

    def get_predicted_evaluation(self, selected):
        selected = get_bin_representation(selected, self.config.OBJ_NUM)
        return self.heuristic.evaluate_selection(selected)

    def select_object_to_try(self, selected):
        selected = get_object_indices(selected)
        # go through the objects and select the one that maximizes the expected information gain
        if self.config.THRESH_ESTIMATION_STRATEGY == "density":
            return self._density_selection(selected)
        elif self.config.THRESH_ESTIMATION_STRATEGY == "random":
            to_choose = set(selected) - set(self.heuristic.locked_subsolution)
            return np.random.choice(list(to_choose))
        elif self.config.THRESH_ESTIMATION_STRATEGY == "intervals":
            return self._interval_selection(selected)
        elif self.config.THRESH_ESTIMATION_STRATEGY == "greedy":
            return self._greedy_selection(selected)
        else:
            raise ValueError(
                f"Unknown threshold estimation strategy: {self.config.THRESH_ESTIMATION_STRATEGY}"
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
        best_score = np.inf
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
        best_score = -np.inf
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
            if f.task_type != obj.task_type:
                continue
            sim = self.environment.get_latent_similarity(obj, f)
            self.threshold_lower_bound = max(self.threshold_lower_bound, sim)
        if self.config.VERBOSITY > 0:
            print(
                f"Lower bound: {self.threshold_lower_bound}, upper bound: {self.threshold_upper_bound}, "
                f"real threshold: {self.config.SIMILARITY_THRESHOLD}"
            )
        self.heuristic.update_threshold_estimate(
            (self.threshold_lower_bound + self.threshold_upper_bound) / 2
        )


if __name__ == "__main__":
    results = {}
    results_threshold_known = {}

    methods = ["density", "random", "intervals", "greedy"]
    heuristics = get_all_heuristics()

    for method in methods:
        config = Config()
        config.THRESH_ESTIMATION_STRATEGY = method
        config.TASK_TYPES = ["sample task"]
        config.VERBOSITY = 0
        single_results = evaluate_all_heuristics(VariableThresholdSolver, config, n=200)
        for name, mean, std, time_taken in single_results:
            results[(name, method)] = (mean, std, time_taken)
        config.USE_REAL_THRESHOLD = True
        single_results = evaluate_all_heuristics(VariableThresholdSolver, config, n=200)
        for name, mean, std, time_taken in single_results:
            results_threshold_known[(name, method)] = (mean, std, time_taken)

    # report the average per heuristic
    for heuristic in heuristics:
        avg = np.mean([results[(heuristic.__name__, method)][0] for method in methods])
        avg_known = np.mean(
            [
                results_threshold_known[(heuristic.__name__, method)][0]
                for method in methods
            ]
        )
        print(f"{heuristic.__name__}: {avg}, {avg_known}")

    # report the average per method
    for method in methods:
        avg = np.mean(
            [results[(heuristic.__name__, method)][0] for heuristic in heuristics]
        )
        avg_known = np.mean(
            [
                results_threshold_known[(heuristic.__name__, method)][0]
                for heuristic in heuristics
            ]
        )
        print(f"{method}: {avg}, {avg_known}")
