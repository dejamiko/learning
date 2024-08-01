from scipy.stats import linregress

from config import Config
from optim.approx_solver import ApproximationSolver
from optim.solver import evaluate_all_heuristics
from utils import (
    ObjectSelectionStrategy as EstStg,
    get_object_indices,
)


class AffineApproximationSolver(ApproximationSolver):
    def __init__(self, config, heuristic_class):
        # it only makes sense to use the real-valued version here
        config.SUCCESS_RATE_BOOLEAN = False
        super().__init__(config, heuristic_class)
        self.affine_function = (1, 0)  # 1 * x + 0

    def _update_state(self, obj_selected):
        self._update_affine_function(obj_selected)
        self.env.update_visual_similarities(self.affine_function)

    def _init_data(self, i):
        super()._init_data(i)
        self._reset_function()

    def _reset_function(self):
        self.affine_function = (1, 0)

    def _select_object_to_try(self, selected):
        selected = get_object_indices(selected)
        print("selected", selected)
        # go through the objects and select the one that maximizes the expected information gain
        if self.config.OBJECT_SELECTION_STRATEGY == EstStg.RANDOM:
            to_choose = set(selected) - set(self.heuristic.locked_subsolution)
            return self._rng.choice(list(to_choose))
        else:
            raise ValueError(
                f"Unknown threshold estimation strategy: {self.config.OBJECT_SELECTION_STRATEGY}"
            )

    def _update_affine_function(self, obj_ind):
        transfer_rates = []
        visual_sims = []
        for o in self.objects:
            if o.task != self.objects[obj_ind].task:
                continue
            # does this make sense? It basically mimics trying the transfer 10 times so it should be sound
            transfer_rates.append(
                self.env._get_real_transfer_probability(obj_ind, o.index)
            )
            visual_sims.append(self.env.get_visual_similarity(obj_ind, o.index))

        # find the best fit function for this data
        slope, intercept, r, p, std_err = linregress(visual_sims, transfer_rates)
        self.affine_function = (
            self.config.MERGING_FACTOR * slope
            + (1 - self.config.MERGING_FACTOR) * self.affine_function[0],
            self.config.MERGING_FACTOR * intercept
            + (1 - self.config.MERGING_FACTOR) * self.affine_function[1],
        )
        print(self.affine_function, r, p, std_err)


if __name__ == "__main__":
    c = Config()
    c.OBJECT_SELECTION_STRATEGY = EstStg.RANDOM

    results = evaluate_all_heuristics(AffineApproximationSolver, c, n=1)
    for name, mean, std, total_time in results:
        print(f"{name}: {mean} +/- {std}, time: {total_time}")
