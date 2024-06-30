import numpy as np


class Object:
    """
    The basic class for objects in the environment. Each object has a latent representation, a visible representation,
    and a task type.
    """

    def __init__(self, index, latent_representation, task_type, c):
        """
        Initialise the object
        :param index: The index of the object
        :param latent_representation: The latent representation of the object
        :param task_type: The task type of the object
        :param c: The configuration object
        """
        assert isinstance(
            latent_representation, np.ndarray
        ), f"Expected np.ndarray, got {type(latent_representation)}"
        assert (
            len(latent_representation.shape) == 1
        ), f"Expected 1D array, got {len(latent_representation.shape)}D"
        assert (
            latent_representation.shape[0] == c.LATENT_DIM
        ), f"Expected array of length {c.LATENT_DIM}, got {latent_representation.shape[0]}"

        self.c = c
        self.name = f"Object {index}"
        self.index = index
        self.latent_repr = latent_representation
        self.visible_repr = self.create_visible_representation()
        self.task_type = task_type

    def create_visible_representation(self):
        """
        Create a visible representation of the object. This is done by adding some noise to the latent representation
        :return: The visible representation of the object
        """
        return self.latent_repr + np.random.normal(
            0, self.c.VISIBLE_REPRESENTATION_NOISE, self.latent_repr.shape
        )

    def get_visual_similarity(self, other):
        """
        Get the similarity between the visible representations of this object and another object
        :param other: The other object
        :return: The similarity between the visible representations. Implemented as the cosine similarity
        """
        return np.dot(self.visible_repr, other.visible_repr) / (
            np.linalg.norm(self.visible_repr) * np.linalg.norm(other.visible_repr)
        )

    def get_latent_similarity(self, other):
        """
        Get the similarity between the latent representations of this object and another object
        :param other: The other object
        :return: The similarity between the latent representations. Implemented as the cosine similarity
        """
        return np.dot(self.latent_repr, other.latent_repr) / (
            np.linalg.norm(self.latent_repr) * np.linalg.norm(other.latent_repr)
        )

    def get_task_type_correspondence(self, other):
        """
        Get the task type correspondence between this object and another object
        :param other: The other object
        :return: True if the task types are the same and False otherwise
        """
        return self.task_type == other.task_type

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"{self.name} ({self.latent_repr}), {self.visible_repr}, {self.task_type}"
        )
