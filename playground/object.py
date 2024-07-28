from abc import ABC, abstractmethod

import numpy as np

from playground.task_types import Task


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

        assert task in [
            t.value for t in Task
        ], f"The task provided {task} is not a valid Task {Task}"

        self.task = task

    @abstractmethod
    def get_visual_similarity(self, other):
        """
        Get the similarity between the visible representations of this object and another object
        :param other: The other object
        :return: The similarity between the visible representations.
        """
        pass  # pragma: no cover

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
