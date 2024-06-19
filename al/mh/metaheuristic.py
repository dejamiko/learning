import multiprocessing
from abc import ABC, abstractmethod

import numpy as np

from playground.environment import Environment
from playground.object import TrajectoryObject


class MetaHeuristic(ABC):
    def __init__(self, c, threshold=None):
        self.c = c

        self._similarity_dict = {}
        self.best_selection = None
        if threshold is None:
            threshold = c.SIMILARITY_THRESHOLD
        self.threshold = threshold
        self.count = 0
        self.locked_subsolution = []
        self.environment = None

    def evaluate_selection_with_constraint_penalty(self, selected):
        constraint_penalty = 0
        if np.sum(selected) > self.c.KNOWN_OBJECT_NUM:
            constraint_penalty = -np.sum(selected) * 100
        return self.evaluate_selection(selected) + constraint_penalty

    def evaluate_selection(self, selected):
        objects = set()
        for i, s in enumerate(selected):
            if s == 0:
                continue
            # assume similarities[s] returns a sorted list of similarities to all objects
            # where o is an object in objects and s is an object in selected
            # use binary search to find the first object with similarity below threshold
            objs, sim = self._similarity_dict[i]
            # sim is a sorted list of similarities, use bin search to find the first object with similarity
            # below threshold
            ind = np.searchsorted(sim, self.threshold)
            # print(f"Selected: {s}, threshold: {c.SIMILARITY_THRESHOLD}, ind: {ind}")
            objects.update(objs[ind:])
        self.count += 1
        return len(objects)

    def initialise_data(self):
        self.environment = Environment(self.c)
        self._similarity_dict = self.environment.generate_objects_ail(TrajectoryObject)
        self.count = 0
        self.locked_subsolution = []
        self.best_selection = None

    def _run_strategy_with_timer(self):
        # TODO fix this because the child object for some reason cannot send back the best selection
        p = multiprocessing.Process(target=self.strategy)
        p.start()
        p.join(20)
        if p.is_alive():
            print("Terminating due to time limit")
            p.terminate()
            p.join()

    def _get_random_initial_selection(self):
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

    def update_threshold_estimate(self, threshold):
        self.threshold = threshold

    def lock_object(self, object_index):
        self.locked_subsolution.append(object_index)

    def get_objects(self):
        return self.environment.get_objects()

    def get_environment(self):
        return self.environment
