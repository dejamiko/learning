import numpy as np
from scipy.stats import linregress

from config import Config
from optim.approx_solver import ApproximationSolver
from optim.mh import get_all_heuristics
from optim.solver import evaluate_all_heuristics
from tm_utils import (
    ObjectSelectionStrategyAffine as EstStg,
    get_object_indices,
    Task,
)


class AffineApproximationSolver(ApproximationSolver):
    def __init__(self, config, heuristic_class):
        # it only makes sense to use the real-valued version here
        config.SUCCESS_RATE_BOOLEAN = False
        super().__init__(config, heuristic_class)
        self.affine_functions = {t: (1.0, 0.0) for t in Task}

    def _update_state(self, obj_selected):
        self._update_affine_functions(obj_selected)
        self.env.update_visual_similarities(self.affine_functions)

    def _init_data(self, i):
        super()._init_data(i)
        self._reset_affine_functions()

    def _reset_affine_functions(self):
        self.affine_functions = {t: (1.0, 0.0) for t in Task}

    def _select_object_to_try(self, selected):
        selected = get_object_indices(selected)
        to_search = list(set(selected) - set(self.heuristic.locked_subsolution))
        # go through the objects and select the one that maximizes the expected information gain
        match self.config.OBJECT_SELECTION_STRATEGY_A:
            case EstStg.RANDOM:
                return self._rng.choice(to_search)
            case EstStg.GREEDY_P:
                return self._select_greedy_p(to_search)
            case EstStg.GREEDY_R:
                return self._select_greedy_r(to_search)
        raise ValueError(
            f"Unknown object selection strategy for affine solver: `{self.config.OBJECT_SELECTION_STRATEGY_A}`"
        )

    def _update_affine_functions(self, obj_ind):
        slope, intercept, p, r = self._get_linregress_for_ind(obj_ind)
        obj_task = self.objects[obj_ind].task
        prev_slope = self.affine_functions[obj_task][0]
        prev_intercept = self.affine_functions[obj_task][1]
        self.affine_functions[obj_task] = (
            float(
                self.config.MERGING_FACTOR * slope
                + (1 - self.config.MERGING_FACTOR) * prev_slope
            ),
            float(
                self.config.MERGING_FACTOR * intercept
                + (1 - self.config.MERGING_FACTOR) * prev_intercept
            ),
        )
        if self.config.VERBOSITY > 0:
            print(f"With function {self.affine_functions}, r={r}, p={p}")

    def _get_linregress_for_ind(self, obj_ind):
        transfer_rates = []
        visual_sims = []
        for o in self.objects:
            if o.task != self.objects[obj_ind].task:
                continue
            # does this make sense? It basically mimics trying the transfer 10 times so it should be sound
            transfer_rates.append(
                self.env.get_real_transfer_probability(obj_ind, o.index)
            )
            visual_sims.append(self.env.get_visual_similarity(obj_ind, o.index))
        # find the best fit function for this data
        slope, intercept, r, p, std_err = linregress(visual_sims, transfer_rates)
        return slope, intercept, p, r

    def _select_greedy_p(self, to_search):
        best_ind = None
        best_p = None
        for ind in to_search:
            _, _, p, _ = self._get_linregress_for_ind(ind)
            if best_p is None or p < best_p:
                best_p = p
                best_ind = ind
        return best_ind

    def _select_greedy_r(self, to_search):
        best_ind = None
        best_r = None
        for ind in to_search:
            _, _, _, r = self._get_linregress_for_ind(ind)
            if best_r is None or r > best_r:
                best_r = r
                best_ind = ind
        return best_ind


if __name__ == "__main__":
    results = {}

    methods = [m for m in EstStg]
    heuristics = get_all_heuristics()

    for method in methods:
        config = Config()
        # config.MH_TIME_BUDGET = 0.1
        config.MH_BUDGET = 5000
        config.OBJECT_SELECTION_STRATEGY_A = method
        config.VERBOSITY = 0
        config.USE_REAL_THRESHOLD = False
        single_results = evaluate_all_heuristics(AffineApproximationSolver, config, n=1)
        for name, mean, std, time_taken in single_results:
            results[(name, method)] = (mean, std, time_taken)

    # report the average per heuristic
    for heuristic in heuristics:
        avg = np.mean([results[(heuristic.__name__, method)][0] for method in methods])
        print(f"{heuristic.__name__}: {avg}")

    # report the average per method
    for method in methods:
        avg = np.mean(
            [results[(heuristic.__name__, method)][0] for heuristic in heuristics]
        )
        print(f"{method.value}: {avg}")
