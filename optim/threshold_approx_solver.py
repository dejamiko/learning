import numpy as np

from config import Config
from optim.approx_solver import ApproximationSolver
from optim.mh import get_all_heuristics
from optim.solver import evaluate_all_heuristics
from tm_utils import (
    ObjectSelectionStrategyThreshold as EstStg,
    get_object_indices,
    Task,
)


class ThresholdApproximationSolver(ApproximationSolver):
    def __init__(self, config, heuristic_class):
        super().__init__(config, heuristic_class)
        # it only makes sense to use the boolean version here
        config.SUCCESS_RATE_BOOLEAN = True
        # self.threshold_lower_bound = 0.0
        # self.threshold_upper_bound = 1.0
        self.threshold_upper_bounds = {t: 1.0 for t in Task}
        self.threshold_lower_bounds = {t: 0.0 for t in Task}

    def _select_object_to_try(self, selected):
        selected = get_object_indices(selected)
        to_search = list(set(selected) - set(self.heuristic.locked_subsolution))
        # go through the objects and select the one that maximizes the expected information gain
        match self.config.OBJECT_SELECTION_STRATEGY_T:
            case EstStg.DENSITY:
                return self._density_selection(to_search)
            case EstStg.RANDOM:
                return self._rng.choice(to_search)
            case EstStg.INTERVALS:
                return self._interval_selection(to_search)
            case EstStg.GREEDY:
                return self._greedy_selection(to_search)
        raise ValueError(
            f"Unknown object selection strategy for threshold solver: `{self.config.OBJECT_SELECTION_STRATEGY_T}`"
        )

    def _update_state(self, obj_selected):
        self._update_bounds(obj_selected)
        self.env.update_visual_sim_thresholds(
            {
                t: (self.threshold_upper_bounds[t] + self.threshold_lower_bounds[t]) / 2
                for t in Task
            }
        )

    def _init_data(self, i):
        super()._init_data(i)
        self._reset_bounds()
        self.config.SIMILARITY_THRESHOLDS = self._rng.uniform(0.3, 0.9, len(Task))
        self.env.update_visual_sim_thresholds(
            {
                t: (self.threshold_upper_bounds[t] + self.threshold_lower_bounds[t]) / 2
                for t in Task
            }
        )

    def _reset_bounds(self):
        if self.config.USE_REAL_THRESHOLD:
            self.threshold_lower_bounds = {
                t: self.config.SIMILARITY_THRESHOLDS[i] for i, t in enumerate(Task)
            }
            self.threshold_upper_bounds = {
                t: self.config.SIMILARITY_THRESHOLDS[i] for i, t in enumerate(Task)
            }
        else:
            self.threshold_upper_bounds = {t: 1.0 for t in Task}
            self.threshold_lower_bounds = {t: 0.0 for t in Task}

    def _density_selection(self, to_search):
        # this can be estimated by selecting one that has the most objects that have a similarity between the bounds
        best_object_index = self._rng.choice(to_search)
        best_score = 0
        for s in to_search:
            s_task = self.objects[s].task
            score = 0
            for o in self.objects:
                if (
                    self.threshold_lower_bounds[s_task]
                    < self.env.get_visual_similarity(s, o.index)
                    < self.threshold_upper_bounds[s_task]
                ) and s_task == o.task:
                    score += 1
            if score > best_score:
                best_score = score
                best_object_index = s
        return best_object_index

    def _interval_selection(self, to_search):
        best_object_index = self._rng.choice(to_search)
        best_score = np.inf
        for s in to_search:
            s_task = self.objects[s].task
            sim_objects_between_bounds = []
            for o in self.objects:
                if (
                    self.threshold_lower_bounds[s_task]
                    < self.env.get_visual_similarity(s, o.index)
                    < self.threshold_upper_bounds[s_task]
                ) and s_task == o.task:
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

    def _greedy_selection(self, to_search):
        # find the selected object that has the most objects that are below the estimated threshold
        best_object_index = self._rng.choice(to_search)
        best_score = -np.inf
        for s in to_search:
            s_task = self.objects[s].task
            score = 0
            for o in self.objects:
                if (
                    self.env.get_visual_similarity(s, o.index)
                    < (
                        self.threshold_lower_bounds[s_task]
                        + self.threshold_upper_bounds[s_task]
                    )
                    / 2
                ):
                    score += 1
            if score > best_score:
                best_score = score
                best_object_index = s
        return best_object_index

    def _update_bounds(self, obj_ind):
        success_indices = []
        failure_indices = []
        obj_task = self.objects[obj_ind].task
        for o in self.objects:
            if o.task != obj_task:
                continue
            if self.env.get_transfer_success(obj_ind, o.index):
                success_indices.append(o.index)
            else:
                failure_indices.append(o.index)
        for s_i in success_indices:
            sim = self.env.get_visual_similarity(obj_ind, s_i)
            self.threshold_upper_bounds[obj_task] = min(
                self.threshold_upper_bounds[obj_task], sim
            )
        for f_i in failure_indices:
            sim = self.env.get_visual_similarity(obj_ind, f_i)
            self.threshold_lower_bounds[obj_task] = max(
                self.threshold_lower_bounds[obj_task], sim
            )
        if self.config.VERBOSITY > 0:
            print(
                f"Lower bounds: {self.threshold_lower_bounds}, upper bounds: {self.threshold_upper_bounds}, "
                f"estimate {[(self.threshold_upper_bounds[t] + self.threshold_lower_bounds[t]) / 2 for t in Task]}, "
                f"real thresholds: {self.config.SIMILARITY_THRESHOLDS}"
            )


if __name__ == "__main__":
    results = {}
    results_threshold_known = {}

    methods = [m for m in EstStg]
    heuristics = get_all_heuristics()

    for method in methods:
        config = Config()
        # config.MH_TIME_BUDGET = 0.1
        config.MH_BUDGET = 5000
        config.OBJECT_SELECTION_STRATEGY_T = method
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
