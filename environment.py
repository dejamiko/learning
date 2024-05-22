import abc

import numpy as np

import config as c


class Object:
    """
    The basic class for objects in the environment. Each object has a latent representation, a visible representation,
    and a task type.
    """

    def __init__(self, index, latent_representation, task_type):
        """
        Initialise the object
        :param index: The index of the object
        :param latent_representation: The latent representation of the object
        :param task_type: The task type of the object
        """
        assert isinstance(latent_representation, np.ndarray), f"Expected np.ndarray, got {type(latent_representation)}"
        assert len(latent_representation.shape) == 1, f"Expected 1D array, got {len(latent_representation.shape)}D"
        assert latent_representation.shape[
                   0] == c.LATENT_DIM, f"Expected array of length {c.LATENT_DIM}, got {latent_representation.shape[0]}"

        self.name = f"Object {index}"
        self.index = index
        self._latent_representation = latent_representation
        self.visible_representation = self.create_visible_representation()
        self.task_type = task_type
        self.demo = None

    def create_visible_representation(self):
        """
        Create a visible representation of the object. This is done by adding some noise to the latent representation
        :return: The visible representation of the object
        """
        return self._latent_representation + np.random.normal(0, c.VISIBLE_REPRESENTATION_NOISE,
                                                              self._latent_representation.shape)

    def get_visual_similarity(self, other):
        """
        Get the similarity between the visible representations of this object and another object
        :param other: The other object
        :return: The similarity between the visible representations. Implemented as the cosine similarity
        """
        return np.dot(self.visible_representation, other.visible_representation) / (
                np.linalg.norm(self.visible_representation) * np.linalg.norm(other.visible_representation))

    def get_latent_similarity(self, other):
        """
        Get the similarity between the latent representations of this object and another object
        :param other: The other object
        :return: The similarity between the latent representations. Implemented as the dot product
        """
        return np.dot(self._latent_representation, other._latent_representation)

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
        return f"{self.name} ({self._latent_representation}), {self.visible_representation}, {self.task_type}, {self.demo}"

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
        return np.allclose(position, self.get_position(), atol=c.POSITION_TOLERANCE)

    def get_position(self):
        """
        Returns the "position" of the object which is some function of the latent representation and task type.
        For now, it is just the latent representation multiplied by the task type
        :return: The "position" of the object
        """
        return self._latent_representation * (c.TASK_TYPES.index(self.task_type) + 1)

    def get_demo(self):
        return self.get_position()


class TrajectoryObject(Object):
    """
    An object that can be interacted with by providing a trajectory of actions

    This assumes the frame is object-centric so there is no need to provide the start pose
    """

    def __init__(self, index, latent_representation, task_type):
        """
        Initialise the object

        :param index: The object index
        :param latent_representation: The latent representation of the object
        :param task_type: The task type of the object
        """
        super().__init__(index, latent_representation, task_type)

    def try_trajectory(self, trajectory):
        """
        Try a trajectory of actions
        :param trajectory: The trajectory to try
        :return: True if the trajectory is successful and False otherwise
        """
        pos = np.sum(trajectory, axis=0)
        return self.check_position(pos)

    def get_demo(self):
        actions = generate_trajectory(self.get_position())
        return actions


def generate_trajectory(position):
    num_steps = max(10, np.ceil(np.linalg.norm(position) / c.MAX_ACTION * 2).astype(int))
    traj = np.linspace(np.zeros_like(position), position, num_steps)
    actions = np.diff(traj, axis=0)
    for i in range(actions.shape[0] - 1):
        noise = np.random.normal(0, c.ACTION_EXPLORATION_DEVIATION, actions[i].shape)
        actions[i] += noise
        actions[i + 1] -= noise
    assert np.allclose(np.sum(actions, axis=0), position, atol=c.POSITION_TOLERANCE)
    return actions


class Oracle(abc.ABC):
    """
    An abstract class for an oracle that can provide demonstrations

    The oracle has a cost associated with using it and can provide a demonstration for an object
    """

    def __init__(self, cost):
        self.cost_per = cost
        self.total_cost = 0

    def get_final_cost(self):
        return self.total_cost

    def reset_cost(self):
        self.total_cost = 0

    @abc.abstractmethod
    def get_demo(self, obj):
        pass


class NoisyOracle(Oracle):
    def get_demo(self, obj):
        self.total_cost += self.cost_per
        obj_demo = obj.get_demo()
        return obj_demo + np.random.normal(0, c.DEMO_NOISE, obj_demo.shape)


class PerfectOracle(Oracle):
    def get_demo(self, obj):
        self.total_cost += self.cost_per
        return obj.get_demo()
