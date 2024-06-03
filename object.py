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
        assert isinstance(latent_representation, np.ndarray), f"Expected np.ndarray, got {type(latent_representation)}"
        assert len(latent_representation.shape) == 1, f"Expected 1D array, got {len(latent_representation.shape)}D"
        assert latent_representation.shape[
                   0] == c.LATENT_DIM, f"Expected array of length {c.LATENT_DIM}, got {latent_representation.shape[0]}"

        self.c = c
        self.name = f"Object {index}"
        self.index = index
        self._latent_repr = latent_representation
        self.visible_repr = self.create_visible_representation()
        self.task_type = task_type

    def create_visible_representation(self):
        """
        Create a visible representation of the object. This is done by adding some noise to the latent representation
        :return: The visible representation of the object
        """
        return self._latent_repr + np.random.normal(0, self.c.VISIBLE_REPRESENTATION_NOISE,
                                                    self._latent_repr.shape)

    def get_visual_similarity(self, other):
        """
        Get the similarity between the visible representations of this object and another object
        :param other: The other object
        :return: The similarity between the visible representations. Implemented as the cosine similarity
        """
        return np.dot(self.visible_repr, other.visible_repr) / (
                np.linalg.norm(self.visible_repr) * np.linalg.norm(other.visible_repr))

    def get_latent_similarity(self, other):
        """
        Get the similarity between the latent representations of this object and another object
        :param other: The other object
        :return: The similarity between the latent representations. Implemented as the dot product
        """
        return np.dot(self._latent_repr, other._latent_repr)

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
        return f"{self.name} ({self._latent_repr}), {self.visible_repr}, {self.task_type}, {self.demo}"

    def try_action(self, action):
        """
        A super simple action which is just trying to predict the "final position" of the object which is some function
        of the latent representation and task type
        """
        return self.check_position(action)

    def check_position(self, position):
        """
        Check if the position is close to the "position" of the object
        :param position: The position to check
        :return: True if the position is close to the "position" of the object
        """
        return np.allclose(position, self.get_position(), atol=self.c.POSITION_TOLERANCE)

    def get_position(self):
        """
        Returns the "position" of the object which is some function of the latent representation and task type.
        For now, it is just the latent representation multiplied by the task type
        :return: The "position" of the object
        """
        return self._latent_repr


class TrajectoryObject(Object):
    """
    An object that can be interacted with by providing a trajectory of actions

    This assumes the frame is object-centric so there is no need to provide the start pose
    """

    def __init__(self, index, latent_representation, task_type, c):
        """
        Initialise the object with a latent representation and a task type

        :param index: The object index
        :param latent_representation: The latent representation of the object
        :param task_type: The task type of the object
        :param c: The configuration object
        """
        super().__init__(index, latent_representation, task_type, c)
        self.waypoints = []

    def generate_waypoints(self):
        """
        Generate waypoints the object needs to pass through
        """
        above = np.zeros_like(self.get_position())
        above[-1] = 0.1
        next_to = np.zeros_like(self.get_position())
        next_to[-1] = 0.2
        if self.task_type == "gripping":
            # for gripping, pick a waypoint above the object, then the object, and back above the object
            self.waypoints = [self.get_position() + above, self.get_position(), self.get_position() + above]
        elif self.task_type == "pushing":
            # for pushing, pick a waypoint at the object position, then a waypoint further away
            self.waypoints = [self.get_position(), self.get_position() + next_to]
        elif self.task_type == "inserting":
            # for inserting, pick a waypoint above the object, then at the object position
            self.waypoints = [self.get_position() + above, self.get_position()]
        self.waypoints = np.array(self.waypoints)
