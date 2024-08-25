import numpy as np
from scipy.stats import linregress

from config import Config
from optim.approx_solver import ApproximationSolver
from optim.mh import EvolutionaryStrategy, RandomisedHillClimbing, TabuSearch
from optim.solver import evaluate_provided_heuristics
from tm_utils import (
    ObjectSelectionStrategyAffine as EstStg,
    get_object_indices,
    Task,
    get_rng,
    ImageEmbeddings,
    SimilarityMeasure,
    ImagePreprocessing,
    NNImageEmbeddings,
    NNSimilarityMeasure,
)


class AffineApproximationSolver(ApproximationSolver):
    def __init__(self, config, heuristic_class):
        # it only makes sense to use the real-valued version here
        config.SUCCESS_RATE_BOOLEAN = False
        super().__init__(config, heuristic_class)
        if config.DO_NOT_ITER:
            self.affine_functions = {
                t: config.AFFINE_FNS[i] for i, t in enumerate(Task)
            }
        else:
            self.affine_functions = {t: (1.0, 0.0) for t in Task}

    def _update_state(self, obj_selected):
        if not self.config.DO_NOT_ITER:
            self._update_affine_functions(obj_selected)
            self.env.update_visual_similarities(self.affine_functions)

    def _init_data(self, i):
        super()._init_data(i)
        self._reset_affine_functions()
        self.env.update_visual_similarities(self.affine_functions)

    def _reset_affine_functions(self):
        if self.config.DO_NOT_ITER:
            self.affine_functions = {
                t: self.config.AFFINE_FNS[i] for i, t in enumerate(Task)
            }
        else:
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


def run_one_affine(run_num, bgt_b, bgt_d):
    c = Config()
    c.MH_TIME_BUDGET = 1
    c.MH_BUDGET = bgt_b
    c.DEMONSTRATION_BUDGET = bgt_d
    c.OBJ_NUM = 51
    c.VERBOSITY = 0
    rng = get_rng(1)
    affine_fns = [
        [
            [
                (0.5 + rng.random(), rng.random()),
                (0.5 + rng.random(), rng.random()),
                (0.5 + rng.random(), rng.random()),
            ],
            [
                (0.5 + rng.random(), rng.random()),
                (0.5 + rng.random(), rng.random()),
                (0.5 + rng.random(), rng.random()),
            ],
            [
                (0.5 + rng.random(), rng.random()),
                (0.5 + rng.random(), rng.random()),
                (0.5 + rng.random(), rng.random()),
            ],
        ],  # Random
        [
            [
                [0.68248846, 0.05567315],
                [0.68248846, 0.05567315],
                [0.68248846, 0.05567315],
            ],
            [
                [0.64965973, 0.08081283],
                [0.64965973, 0.08081283],
                [0.64965973, 0.08081283],
            ],
            [
                [0.83972853, 0.06214219],
                [0.83972853, 0.06214219],
                [0.83972853, 0.06214219],
            ],
        ],  # Reasonable
        [
            [
                [0.7181944867911111, 0.1138801077232438],
                [0.7687713724858166, -0.04494050594772425],
                [0.5604995163692343, 0.09807984839240697],
            ],
            [
                [0.7227911796537266, 0.12563668556229102],
                [0.703188388050649, 0.0025273010088077696],
                [0.5229996181595288, 0.11427449671574019],
            ],
            [
                [0.9498635218251053, 0.018646840409686005],
                [0.87386853069105, 0.03705817606803352],
                [0.6954535244565685, 0.13072154780451728],
            ],
        ],  # Good
    ]
    configs = [
        (
            [],
            ImageEmbeddings.DINO_FULL,
            SimilarityMeasure.COSINE,
            False,
        ),
        (
            [ImagePreprocessing.CROPPING, ImagePreprocessing.BACKGROUND_REM],
            ImageEmbeddings.DINO_FULL,
            SimilarityMeasure.COSINE,
            True,
        ),
        (
            [
                ImagePreprocessing.CROPPING,
                ImagePreprocessing.BACKGROUND_REM,
                ImagePreprocessing.GREYSCALE,
            ],
            NNImageEmbeddings.SIAMESE,
            NNSimilarityMeasure.TRAINED,
            True,
        ),
    ]
    failures_1 = 0
    failures_2 = 0
    for ind, (ps, emb, sim, use_all) in enumerate(configs):
        c.IMAGE_PREPROCESSING = ps
        c.IMAGE_EMBEDDINGS = emb
        c.SIMILARITY_MEASURE = sim
        c.USE_ALL_IMAGES = use_all
        max_scores = []
        for f in affine_fns:
            c.AFFINE_FNS = f[ind]
            c.DO_NOT_ITER = True
            results = evaluate_provided_heuristics(
                AffineApproximationSolver,
                c,
                n=run_num,
                heuristics=(
                    EvolutionaryStrategy,
                    RandomisedHillClimbing,
                    TabuSearch,
                ),
            )
            max_scores.append(max([r[1] for r in results]))
            # print(f"For config {c}")
            # for name, mean, std, total_time in results:
            #     print(f"{name}: {mean}")
        for i in range(len(max_scores) - 1):
            if max_scores[i + 1] < max_scores[i]:
                failures_1 += 1
        c.DO_NOT_ITER = False
        other_scores_min = 51
        for strat in EstStg:
            c.OBJECT_SELECTION_STRATEGY_A = strat
            results = evaluate_provided_heuristics(
                AffineApproximationSolver,
                c,
                n=run_num,
                heuristics=(
                    EvolutionaryStrategy,
                    RandomisedHillClimbing,
                    TabuSearch,
                ),
            )
            max_mh = np.max([r[1] for r in results])
            if strat == EstStg.RANDOM:
                random_score = max_mh
            else:
                other_scores_min = min(other_scores_min, max_mh)
            # print(f"For config {c}")
            # for name, mean, std, total_time in results:
            #     print(f"{name}: {mean}")
        if random_score > other_scores_min:
            failures_2 += 1
    return failures_1, failures_2


if __name__ == "__main__":
    for b in [250, 500, 750, 1000, 1500, 2000]:
        for d in [5, 6, 7, 8, 9, 10]:
            f1, f2 = run_one_affine(10, b, d)
            print("For", b, "and", d, "got", f1, f2)

    # results = {}
    #
    # methods = [m for m in EstStg]
    # heuristics = get_all_heuristics()
    #
    # for method in methods:
    #     config = Config()
    #     # config.MH_TIME_BUDGET = 0.1
    #     config.MH_BUDGET = 5000
    #     config.OBJECT_SELECTION_STRATEGY_A = method
    #     config.VERBOSITY = 0
    #     single_results = evaluate_all_heuristics(AffineApproximationSolver, config, n=1)
    #     for name, mean, std, time_taken in single_results:
    #         results[(name, method)] = (mean, std, time_taken)
    #
    # # report the average per heuristic
    # for heuristic in heuristics:
    #     avg = np.mean([results[(heuristic.__name__, method)][0] for method in methods])
    #     print(f"{heuristic.__name__}: {avg}")
    #
    # # report the average per method
    # for method in methods:
    #     avg = np.mean(
    #         [results[(heuristic.__name__, method)][0] for heuristic in heuristics]
    #     )
    #     print(f"{method.value}: {avg}")
