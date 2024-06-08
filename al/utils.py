import multiprocessing
import time
from abc import ABC, abstractmethod

import numpy as np

from environment import Environment
from object import TrajectoryObject


class NeighbourGenerator:
    def __init__(self, object_indices, selected):
        self.selected = selected
        self.not_selected = list(set(object_indices) - set(selected))

    def __iter__(self):
        self.selected = np.random.permutation(self.selected)
        self.not_selected = np.random.permutation(self.not_selected)
        for i in self.selected:
            for j in self.not_selected:
                neighbour = self.selected.copy()
                neighbour[neighbour == i] = j
                yield neighbour


class MetaHeuristic(ABC):
    def __init__(self, c):
        self.c = c
        self._times_taken_on_strategy = []
        self._similarity_dict = {}
        self.objects = None

    def get_cost(self, selected):
        return len(self.objects) - self.evaluate_selection(selected)

    def get_random_neighbour(self, object_ind, selected):
        pool = set(object_ind) - set(selected)
        addition = np.random.choice(list(pool), 1, replace=False)[0]
        removal = np.random.choice(selected, 1, replace=False)[0]
        selected = list(selected)
        selected.remove(removal)
        selected.append(addition)
        return selected

    def evaluate_selection(self, selected_ind):
        objects = set()
        for s in selected_ind:
            # assume similarities[s] returns a sorted list of similarities to all objects
            # where o is an object in objects and s is an object in selected
            # use binary search to find the first object with similarity below threshold
            objs, sim = self._similarity_dict[s]
            # sim is a sorted list of similarities, use bin search to find the first object with similarity
            # below threshold
            ind = np.searchsorted(sim, self.c.SIMILARITY_THRESHOLD)
            # print(f"Selected: {s}, threshold: {c.SIMILARITY_THRESHOLD}, ind: {ind}")
            objects.update(objs[ind:])
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
        self.c.TASK_TYPES = ["gripping"]
        self.c.OBJ_NUM = 100
        self.c.LATENT_DIM = 10
        env = Environment(self.c)
        env.generate_objects_ail(TrajectoryObject)
        similarities = env.get_similarities()
        self._generate_similarity_dict(env, similarities)
        self.objects = env.get_objects()

    def get_mean_time(self):
        return np.mean(self._times_taken_on_strategy)

    def get_total_time(self):
        return np.sum(self._times_taken_on_strategy)

    def _generate_similarity_dict(self, env, similarities):
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

    @abstractmethod
    def strategy(self):
        pass
