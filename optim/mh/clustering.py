from warnings import simplefilter

import numpy as np
from scipy.cluster.hierarchy import ClusterWarning
from sklearn.cluster import AgglomerativeClustering

from config import Config
from optim.mh.metaheuristic import MetaHeuristic
from playground.environment import Environment
from tm_utils import Task

simplefilter("ignore", ClusterWarning)


class ClusteringSearch(MetaHeuristic):
    def __init__(self, c, environment, locked_subsolution):
        super().__init__(c, environment, locked_subsolution)
        self.best_selection = None

    def strategy(self):
        selected = np.zeros(self.c.OBJ_NUM)

        distances = np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM))
        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                if i == j:
                    distances[i, j] = 0
                else:
                    distances[i, j] = 1 - self.environment.storage.get_visual_similarity(i, j)

        model = AgglomerativeClustering(
            n_clusters=self.c.DEMONSTRATION_BUDGET,
            linkage='complete'
        ).fit(distances)

        clusters = {}
        for i, alloc in enumerate(model.labels_):
            if int(alloc) not in clusters:
                clusters[int(alloc)] = []
            clusters[int(alloc)].append(i)
        # find a member of each cluster in a greedy way
        for members in clusters.values():
            max_reachable_count = 0
            best_object_index = -1
            for o in members:
                _, sim = self.environment.similarity_dict[o]
                ind = np.searchsorted(
                    sim, self.c.SIMILARITY_THRESHOLDS[Task.get_ind(self.environment.get_objects()[o].task)]
                )
                if ind > max_reachable_count:
                    max_reachable_count = ind
                    best_object_index = o
            selected[best_object_index] = 1

        self.best_selection = selected
        return self.best_selection

    def get_best_solution(self):
        return self.best_selection


if __name__ == "__main__":
    config = Config()
    env = Environment(config)
    cs = ClusteringSearch(config, env, [])

    selected = cs.strategy()
    print(cs.evaluate_selection_with_constraint_penalty(selected))
