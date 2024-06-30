from abc import ABC, abstractmethod

import numpy as np


class MetaHeuristic(ABC):
    def __init__(self, c, environment, locked_subsolution, threshold=None):
        self.c = c
        if threshold is None:
            threshold = c.SIMILARITY_THRESHOLD
        self.threshold = threshold
        self.count = 0
        self.locked_subsolution = locked_subsolution
        self.environment = environment

    def evaluate_selection_with_constraint_penalty(self, selected):
        constraint_penalty = 0
        if np.sum(selected) > self.c.KNOWN_OBJECT_NUM:
            constraint_penalty = -np.sum(selected) * 100
        return self.evaluate_selection(selected) + constraint_penalty

    def evaluate_selection(self, selected):
        self.count += 1
        if self.c.USE_TRANSFER_EVALUATION:
            return self.environment.evaluate_selection_transfer_based(selected)
        else:
            return self.environment.evaluate_selection_similarity_based(
                selected, self.threshold
            )

    def get_random_initial_selection(self):
        object_indices_not_locked = np.array(
            list(set(np.arange(self.c.OBJ_NUM)) - set(self.locked_subsolution))
        )
        selected = np.zeros(self.c.OBJ_NUM)
        selected[self.locked_subsolution] = 1
        selected[
            np.random.choice(
                object_indices_not_locked,
                self.c.KNOWN_OBJECT_NUM - len(self.locked_subsolution),
                replace=False,
            )
        ] = 1
        return selected

    @abstractmethod
    def strategy(self):
        pass
