import multiprocessing
import time
from abc import ABC, abstractmethod

import numpy as np

from playground.environment import Environment
from playground.object import TrajectoryObject


class NeighbourGenerator:
    def __init__(self, selected, locked_subsolution):
        self.selected = selected
        self.locked_subsolution = locked_subsolution

    def __iter__(self):
        indices = np.random.permutation(np.arange(len(self.selected)))
        indices2 = np.random.permutation(np.arange(len(self.selected)))
        for i in indices:
            if self.selected[i] == 1 and i not in self.locked_subsolution:
                for j in indices2:
                    if self.selected[j] == 0:
                        new_selected = self.selected.copy()
                        new_selected[i] = 0
                        new_selected[j] = 1
                        yield new_selected


class MetaHeuristic(ABC):
    def __init__(self, c, threshold=None):
        self.c = c
        self._times_taken_on_strategy = []
        self._similarity_dict = {}
        self.best_selection = None
        if threshold is None:
            threshold = c.SIMILARITY_THRESHOLD
        self.threshold = threshold
        self.count = 0

        self.locked_subsolution = []

    def get_cost(self, selected):
        return self.c.OBJ_NUM - self.evaluate_selection(selected)

    def get_random_neighbour(self, selected):
        # find a random index where selected is 1 and another where it is 0
        while True:
            i = np.random.choice(np.where(selected == 1)[0])
            if i not in self.locked_subsolution:
                break
        j = np.random.choice(np.where(selected == 0)[0])
        new_selected = selected.copy()
        new_selected[i] = 0
        new_selected[j] = 1
        return new_selected

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

    def evaluate_strategy(self, n=5):
        counts = []
        for i in range(n):
            self._initialise_data(i)
            # let this run for the time budget specified in the config and return the best selection
            start = time.time()
            # self._run_strategy_with_timer()
            selected = self.strategy()
            end = time.time()
            # selected = strategy(objects, c, similarities) # this is the original line
            count = self.evaluate_selection(selected)
            counts.append(count)
            self._times_taken_on_strategy.append(end - start)
        return np.mean(counts), np.std(counts)

    def _initialise_data(self, i):
        np.random.seed(i)
        self.c.TASK_TYPES = ["sample task"]
        self.c.OBJ_NUM = 100
        self.c.LATENT_DIM = 10
        env = Environment(self.c)
        env.generate_objects_ail(TrajectoryObject)
        self.generate_similarity_dict(env)

    def get_mean_time(self):
        return np.mean(self._times_taken_on_strategy)

    def get_total_time(self):
        return np.sum(self._times_taken_on_strategy)

    def generate_similarity_dict(self, env):
        similarities = env.get_similarities()
        for o in env.get_objects():
            s = similarities[o.task_type][o.index]
            ar = []
            for o2 in env.get_objects():
                if o.task_type != o2.task_type:
                    continue
                ar.append((o2.index, s[o2.index]))
            ss = sorted(ar, key=lambda x: x[1])
            self._similarity_dict[o.index] = ([x[0] for x in ss], [x[1] for x in ss])

    def _run_strategy_with_timer(self):
        # TODO fix this because the child object for some reason cannot send back the best selection
        p = multiprocessing.Process(target=self.strategy)
        p.start()
        p.join(20)
        if p.is_alive():
            print("Terminating due to time limit")
            p.terminate()
            p.join()

    def _get_initial_selection(self):
        object_indices = np.array(
            list(set(np.arange(self.c.OBJ_NUM)) - set(self.locked_subsolution))
        )
        selected = np.zeros(self.c.OBJ_NUM)
        selected[self.locked_subsolution] = 1
        selected[
            np.random.choice(
                object_indices,
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


def get_object_indices(selected):
    return np.where(selected == 1)[0]


def get_bin_representation(selected, max_len):
    selected_z = np.zeros(max_len)
    selected_z[selected] = 1
    return selected_z
