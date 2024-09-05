import numpy as np

from config import Config
from optim.approx_solver import ApproximationSolver
from optim.mh import (
    EvolutionaryStrategy,
    RandomisedHillClimbing,
    TabuSearch,
    get_all_heuristics,
)
from optim.solver import evaluate_provided_heuristics, evaluate_all_heuristics
from tm_utils import (
    ObjectSelectionStrategyThreshold as EstStg,
    get_object_indices,
    Task,
    ImageEmbeddings,
    SimilarityMeasure,
    ImagePreprocessing,
    ContourImageEmbeddings,
    ContourSimilarityMeasure,
    NNImageEmbeddings,
    NNSimilarityMeasure,
    get_rng,
)


class ThresholdApproximationSolver(ApproximationSolver):
    def __init__(self, config, heuristic_class):
        super().__init__(config, heuristic_class)
        # it only makes sense to use the boolean version here
        config.SUCCESS_RATE_BOOLEAN = True
        if config.DO_NOT_ITER:
            self.threshold_upper_bounds = {
                t: config.SIMILARITY_THRESHOLDS[i] for i, t in enumerate(Task)
            }
            self.threshold_lower_bounds = {
                t: config.SIMILARITY_THRESHOLDS[i] for i, t in enumerate(Task)
            }
        else:
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
        if not self.config.DO_NOT_ITER:
            self._update_bounds(obj_selected)
            self.env.update_visual_sim_thresholds(
                {
                    t: (self.threshold_upper_bounds[t] + self.threshold_lower_bounds[t])
                    / 2
                    for t in Task
                }
            )

    def _init_data(self, i):
        super()._init_data(i)
        self._reset_bounds()
        self.env.update_visual_sim_thresholds(
            {
                t: (self.threshold_upper_bounds[t] + self.threshold_lower_bounds[t]) / 2
                for t in Task
            }
        )

    def _reset_bounds(self):
        if self.config.DO_NOT_ITER:
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


def run_one_thresh(run_num, bgt_b, bgt_d):
    c = Config()
    c.MH_TIME_BUDGET = 1
    c.MH_BUDGET = bgt_b
    c.DEMONSTRATION_BUDGET = bgt_d
    c.OBJ_NUM = 51
    c.VERBOSITY = 0
    rng = get_rng(1)
    thresholds = [
        [
            rng.random(3).tolist(),
            rng.random(3).tolist(),
            rng.random(3).tolist(),
        ],  # Random
        [
            [0.6214405360134004, 0.6214405360134004, 0.6214405360134004],
            [0.09715242881072028, 0.09715242881072028, 0.09715242881072028],
            [0.5460636515912897, 0.5460636515912897, 0.5460636515912897],
        ],  # Reasonable
        [
            [0.48743718592964824, 0.6331658291457286, 0.7437185929648241],
            [0.04522613065326633, 0.11055276381909548, 0.135678391959799],
            [0.5376884422110553, 0.49246231155778897, 0.6080402010050251],
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
            [ImagePreprocessing.CROPPING, ImagePreprocessing.GREYSCALE],
            ContourImageEmbeddings.MASK_RCNN,
            ContourSimilarityMeasure.ASD,
            True,
        ),
        (
            [ImagePreprocessing.CROPPING, ImagePreprocessing.GREYSCALE],
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
        for t in thresholds:
            c.SIMILARITY_THRESHOLDS = t[ind]
            c.DO_NOT_ITER = True
            results = evaluate_provided_heuristics(
                ThresholdApproximationSolver,
                c,
                n=run_num,
                heuristics=(
                    EvolutionaryStrategy,
                    RandomisedHillClimbing,
                    TabuSearch,
                ),
            )
            max_scores.append(max([r[1] for r in results]))
            print(f"For config {c}")
            for name, mean, std, total_time in results:
                print(f"{name}: {mean}")

        for i in range(len(max_scores) - 1):
            if max_scores[i + 1] < max_scores[i]:
                failures_1 += 1
        c.DO_NOT_ITER = False
        other_scores_min = 51
        for strat in EstStg:
            c.OBJECT_SELECTION_STRATEGY_T = strat
            results = evaluate_provided_heuristics(
                ThresholdApproximationSolver,
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
            print(f"For config {c}")
            for name, mean, std, total_time in results:
                print(f"{name}: {mean}")
        if random_score > other_scores_min:
            failures_2 += 1
    return failures_1, failures_2


if __name__ == "__main__":
    results = {}
    results_threshold_known = {}

    methods = [m for m in EstStg]
    heuristics = get_all_heuristics()

    for method in methods:
        config = Config()
        config.MH_TIME_BUDGET = 0.1
        config.MH_BUDGET = 1000
        config.OBJECT_SELECTION_STRATEGY_T = method
        config.VERBOSITY = 0
        single_results = evaluate_all_heuristics(
            ThresholdApproximationSolver, config, n=1
        )
        for name, mean, std, time_taken in single_results:
            results[(name, method)] = (mean, std, time_taken)
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
