from abc import ABC, abstractmethod

import numpy as np

from tm_utils import Task, SimilarityMeasure


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
            case SimilarityMeasure.EUCLIDEAN_INV:
                return self._get_euclidean_inverse(
                    self.visible_repr, other.visible_repr
                )
            case SimilarityMeasure.EUCLIDEAN_EXP:
                return self._get_euclidean_exponential(
                    self.visible_repr, other.visible_repr, self.c.SIM_MEASURE_SIGMA
                )
            case SimilarityMeasure.MANHATTAN_INV:
                return self._get_manhattan_inverse(
                    self.visible_repr, other.visible_repr
                )
            case SimilarityMeasure.MANHATTAN_EXP:
                return self._get_manhattan_exponential(
                    self.visible_repr, other.visible_repr, self.c.SIM_MEASURE_SIGMA
                )
            case SimilarityMeasure.PEARSON:
                return self._get_pearson(self.visible_repr, other.visible_repr)
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
    def _get_euclidean_inverse(a, b):
        return 1 / (1 + np.linalg.norm(a - b))

    @staticmethod
    def _get_euclidean_exponential(a, b, sigma):
        return np.exp(-np.linalg.norm(a - b) / (2 * sigma**2))

    @staticmethod
    def _get_manhattan_inverse(a, b):
        return 1 / (1 + np.sum(np.abs(a - b)))

    @staticmethod
    def _get_manhattan_exponential(a, b, sigma):
        return np.exp(-np.sum(np.abs(a - b)) / (2 * sigma**2))

    @staticmethod
    def _get_pearson(a, b):
        return np.corrcoef(a, b)[0, 1]
