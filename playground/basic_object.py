import numpy as np

from .object import Object


class BasicObject(Object):
    """
    The basic class for objects in the environment. Each object has a latent representation, a visible representation,
    and a task type.
    """

    def __init__(self, index, c, task, latent_representation, rng):
        """
        Initialise the object
        :param index: The index of the object
        :param c: The configuration object
        :param task: The task type of the object
        :param latent_representation: The latent representation of the object
        """
        super().__init__(index, c, task)
        assert isinstance(
            latent_representation, np.ndarray
        ), f"Expected np.ndarray, got {type(latent_representation)}"
        assert (
            len(latent_representation.shape) == 1
        ), f"Expected 1D array, got {len(latent_representation.shape)}D"
        assert (
            latent_representation.shape[0] == c.LATENT_DIM
        ), f"Expected array of length {c.LATENT_DIM}, got {latent_representation.shape[0]}"

        self.name = f"Object {index}"
        self.latent_repr = latent_representation
        self.visible_repr = self._create_visible_representation(rng)

    def get_latent_similarity(self, other):
        """
        Get the similarity between the latent representations of this object and another object
        :param other: The other object
        :return: The similarity between the latent representations. Implemented as the cosine similarity
        """
        return self._get_cos_sim(self.latent_repr, other.latent_repr)

    def _create_visible_representation(self, rng):
        """
        Create a visible representation of the object. This is done by adding some noise to the latent representation
        :return: The visible representation of the object
        """
        return self.latent_repr + rng.normal(
            0, self.c.VISIBLE_REPRESENTATION_NOISE, self.latent_repr.shape
        )

    def __str__(self):
        return (
            f"{self.name} ({self.latent_repr}), {self.visible_repr}, {self.task.value}"
        )
