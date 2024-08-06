from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist

from tm_utils import Task, SimilarityMeasure, ContourSimilarityMeasure


class Object(ABC):
    """
    The basic class for objects in the environment. Each object has a latent representation, a visible representation,
    and a task type.
    """

    def __init__(self, index, c, task):
        """
        Initialise the object
        :param index: The index of the object
        :param c: The configuration object
        :param task: The task this object is prepared for
        """
        self.index = index
        self.c = c

        self.task = Task(task)
        self.visible_repr = None

    def get_visual_similarity(self, other):
        match self.c.SIMILARITY_MEASURE:
            case SimilarityMeasure.COSINE:
                return self._get_cos_sim(self.visible_repr, other.visible_repr)
            case SimilarityMeasure.EUCLIDEAN:
                return self._get_euclidean(self.visible_repr, other.visible_repr)
            case SimilarityMeasure.MANHATTAN:
                return self._get_manhattan(self.visible_repr, other.visible_repr)
            case SimilarityMeasure.PEARSON:
                return self._get_pearson(self.visible_repr, other.visible_repr)
            case ContourSimilarityMeasure.HAUSDORFF:
                return self._get_hausdorff(self.visible_repr, other.visible_repr)
            case ContourSimilarityMeasure.ASD:
                return self._get_asd(self.visible_repr, other.visible_repr)
        raise ValueError(
            f"Unknown similarity measure provided `{self.c.SIMILARITY_MEASURE}`."
        )

    def __repr__(self):
        """
        This is used for printing collections of objects in a readable way.
        """
        return self.__str__()

    @abstractmethod
    def __str__(self):
        pass  # pragma: no cover

    @staticmethod
    def _get_cos_sim(a, b, eps=1e-8):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

    @staticmethod
    def _get_euclidean(a, b):
        return 1 / (1 + np.linalg.norm(a - b))

    @staticmethod
    def _get_manhattan(a, b):
        return 1 / (1 + np.sum(np.abs(a - b)))

    @staticmethod
    def _get_pearson(a, b):
        return np.corrcoef(a, b)[0, 1]

    @staticmethod
    def _get_hausdorff(a, b):
        hausdorff_dist = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
        return 1 / (1 + hausdorff_dist)

    @staticmethod
    def _get_asd(a, b):
        dist_matrix = cdist(a, b)

        # Calculate the average minimum distance from points1 to points2
        min_distances_1_to_2 = np.min(dist_matrix, axis=1)
        avg_dist_1_to_2 = np.mean(min_distances_1_to_2)

        # Calculate the average minimum distance from points2 to points1
        min_distances_2_to_1 = np.min(dist_matrix, axis=0)
        avg_dist_2_to_1 = np.mean(min_distances_2_to_1)

        # The average surface distance is the mean of these two values
        asd = (avg_dist_1_to_2 + avg_dist_2_to_1) / 2.0
        return 1 / (1 + asd)
