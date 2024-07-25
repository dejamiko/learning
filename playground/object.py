from abc import ABC, abstractmethod


class Object(ABC):
    """
    The basic class for objects in the environment. Each object has a latent representation, a visible representation,
    and a task type.
    """

    def __init__(self, c, task):
        """
        Initialise the object
        :param c: The configuration object
        :param task: The task this object is prepared for
        """
        self.c = c
        self.task = task

    @abstractmethod
    def get_visual_similarity(self, other):
        """
        Get the similarity between the visible representations of this object and another object
        :param other: The other object
        :return: The similarity between the visible representations.
        """
        pass

    def __repr__(self):
        """
        This is used for printing collections of objects in a readable way.
        """
        return self.__str__()

    @abstractmethod
    def __str__(self):
        pass
