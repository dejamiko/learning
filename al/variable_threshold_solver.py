"""
This file contains the Solver class for solving the problem based on the visual similarity of the objects but
the final evaluation is based on the latent similarity of the objects. The threshold for the similarity is not known
and is estimated based on the interactions of the objects.
"""

import numpy as np

from al.sa import SimulatedAnnealing
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
        while len(selected) < c.KNOWN_OBJECT_NUM:
            heuristic_selected = self.heuristic.strategy()
            obj_to_try = self.select_object_to_try(heuristic_selected)
            selected.append(obj_to_try)
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
        # this can be estimated by selecting one that has the most objects that have a similarity between the bounds
        best_object_index = np.random.choice(selected)
        best_score = 0
        for o in self.objects:
            if o.index in selected:
                continue
            score = 0
            for s in selected:
                if (
                        self.threshold_lower_bound
                        < self.environment.get_visual_similarity(self.objects[s], o)
                        < self.threshold_upper_bound
                ) and self.objects[s].task_type == o.task_type:
                    score += 1
            if score > best_score:
                best_score = score
                best_object_index = o.index
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
        if c.VERBOSITY > 0:
            print(
                f"Lower bound: {self.threshold_lower_bound}, upper bound: {self.threshold_upper_bound}, real threshold: {self.config.SIMILARITY_THRESHOLD}"
            )
        self.heuristic.update_threshold_estimate(
            (self.threshold_lower_bound + self.threshold_upper_bound) / 2
        )


if __name__ == "__main__":
    counts = []
    for s in range(50):
        np.random.seed(s)
        c = Config()
        c.TASK_TYPES = ["sample task"]
        c.OBJ_NUM = 100
        c.LATENT_DIM = 10
        c.VERBOSITY = 0
        c.SIMILARITY_THRESHOLD = np.random.uniform(0.5, 0.9)
        solver = Solver(c, SimulatedAnnealing)
        selected = solver.solve()
        count = solver.evaluate(selected)

        np.random.seed(s)
        c = Config()
        c.TASK_TYPES = ["sample task"]
        c.OBJ_NUM = 100
        c.LATENT_DIM = 10
        c.VERBOSITY = 0
        c.SIMILARITY_THRESHOLD = np.random.uniform(0.5, 0.9)
        solver = Solver(c, SimulatedAnnealing, c.SIMILARITY_THRESHOLD, c.SIMILARITY_THRESHOLD)
        selected = solver.solve()
        count2 = solver.evaluate(selected)

        counts.append((count, count2))

    # find the average gain in performance by knowing the threshold
    gain = 0
    total = 0
    for c1, c2 in counts:
        gain += c2 - c1
        total += c2
    gain /= len(counts)
    total /= len(counts)

    print(f"Average gain in performance by knowing the threshold: {gain}")
    # also report this as a percentage of the number of objects successfully transferred
    print(f"Average gain in performance by knowing the threshold as a percentage of the total: {gain / total * 100}%")
