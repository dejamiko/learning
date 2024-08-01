import logging

import numpy as np

from config import Config
from optim.approx_solver import ApproximationSolver
from optim.mh import get_all_heuristics
from optim.solver import evaluate_all_heuristics
from utils import (
    ThresholdEstimationStrategy as EstStg,
    get_object_indices,
)


class ThresholdApproximationSolver(ApproximationSolver):
    def __init__(self, config, heuristic_class):
        super().__init__(config, heuristic_class)
        # it only makes sense to use the boolean version here
        config.SUCCESS_RATE_BOOLEAN = True
        self.threshold_lower_bound = 0.0
        self.threshold_upper_bound = 1.0

    def _select_object_to_try(self, selected):
        selected = get_object_indices(selected)
        # go through the objects and select the one that maximizes the expected information gain
        if self.config.THRESH_ESTIMATION_STRATEGY == EstStg.DENSITY:
            return self._density_selection(selected)
        elif self.config.THRESH_ESTIMATION_STRATEGY == EstStg.RANDOM:
            to_choose = set(selected) - set(self.heuristic.locked_subsolution)
            return self._rng.choice(list(to_choose))
        elif self.config.THRESH_ESTIMATION_STRATEGY == EstStg.INTERVALS:
            return self._interval_selection(selected)
        elif self.config.THRESH_ESTIMATION_STRATEGY == EstStg.GREEDY:
            return self._greedy_selection(selected)
        else:
            raise ValueError(
                f"Unknown threshold estimation strategy: {self.config.THRESH_ESTIMATION_STRATEGY}"
            )

    def _update_state(self, obj_to_try):
        self._update_bounds(obj_to_try)
        self.env.update_visual_sim_threshold(
            (self.threshold_lower_bound + self.threshold_upper_bound) / 2
        )

    def _init_data(self, i):
        super()._init_data(i)
        self._reset_bounds()
        self.config.SIMILARITY_THRESHOLD = self._rng.uniform(0.3, 0.9)
        self.env.update_visual_sim_threshold(
            (self.threshold_lower_bound + self.threshold_upper_bound) / 2
        )

    def _reset_bounds(self):
        if self.config.USE_REAL_THRESHOLD:
            self.threshold_lower_bound = self.config.SIMILARITY_THRESHOLD
            self.threshold_upper_bound = self.config.SIMILARITY_THRESHOLD
        else:
            self.threshold_lower_bound = 0.0
            self.threshold_upper_bound = 1.0

    def _density_selection(self, selected):
        # this can be estimated by selecting one that has the most objects that have a similarity between the bounds
        to_search = list(set(selected) - set(self.heuristic.locked_subsolution))
        best_object_index = self._rng.choice(to_search)
        best_score = 0
        for s in to_search:
            score = 0
            for o in self.objects:
                if (
                    self.threshold_lower_bound
                    < self.env.get_visual_similarity(s, o.index)
                    < self.threshold_upper_bound
                ) and self.objects[s].task == o.task:
                    score += 1
            if score > best_score:
                best_score = score
                best_object_index = s
        return best_object_index

    def _interval_selection(self, selected):
        to_search = list(set(selected) - set(self.heuristic.locked_subsolution))
        best_object_index = self._rng.choice(to_search)
        best_score = np.inf
        for s in to_search:
            sim_objects_between_bounds = []
            for o in self.objects:
                if (
                    self.threshold_lower_bound
                    < self.env.get_visual_similarity(s, o.index)
                    < self.threshold_upper_bound
                ) and self.objects[s].task == o.task:
                    sim_objects_between_bounds.append(
                        self.env.get_visual_similarity(s, o.index)
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
        best_object_index = self._rng.choice(to_search)
        best_score = -np.inf
        for s in to_search:
            score = 0
            for o in self.objects:
                if (
                    self.env.get_visual_similarity(s, o.index)
                    < (self.threshold_lower_bound + self.threshold_upper_bound) / 2
                ):
                    score += 1
            if score > best_score:
                best_score = score
                best_object_index = s
        return best_object_index

    def _update_bounds(self, obj_ind):
        # TODO Work on this taking into account the noisy bound problem
        success_indices = []
        failure_indices = []
        for o in self.objects:
            if o.task != self.objects[obj_ind].task:
                continue
            if self.env.get_transfer_success(obj_ind, o.index):
                success_indices.append(o.index)
            else:
                failure_indices.append(o.index)
        for s_i in success_indices:
            sim = self.env.get_visual_similarity(obj_ind, s_i)
            self.threshold_upper_bound = min(self.threshold_upper_bound, sim)
        for f_i in failure_indices:
            sim = self.env.get_visual_similarity(obj_ind, f_i)
            self.threshold_lower_bound = max(self.threshold_lower_bound, sim)
        if self.config.VERBOSITY > 0:
            print(
                f"Lower bound: {self.threshold_lower_bound}, upper bound: {self.threshold_upper_bound}, "
                f"estimate {(self.threshold_lower_bound + self.threshold_upper_bound) / 2}, "
                f"real threshold: {self.config.SIMILARITY_THRESHOLD}"
            )


if __name__ == "__main__":
    results = {}
    results_threshold_known = {}

    methods = [m for m in EstStg]
    heuristics = get_all_heuristics()

    stopit_logger = logging.getLogger("stopit")
    stopit_logger.setLevel(logging.ERROR)

    for method in methods:
        config = Config()
        # config.MH_TIME_BUDGET = 0.1
        config.MH_BUDGET = 5000
        config.THRESH_ESTIMATION_STRATEGY = method
        config.VERBOSITY = 0
        config.USE_REAL_THRESHOLD = False
        single_results = evaluate_all_heuristics(
            ThresholdApproximationSolver, config, n=1
        )
        for name, mean, std, time_taken in single_results:
            results[(name, method)] = (mean, std, time_taken)
        config.USE_REAL_THRESHOLD = True
        single_results = evaluate_all_heuristics(
            ThresholdApproximationSolver, config, n=1
        )
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
        print(f"{method.value}: {avg}, {avg_known}")
