import numpy as np

from playground.storage import ObjectStorage


class Environment:
    """
    A class to represent the environment. It provides an interface to the solver which abstracts the storage, oracle,
    and objects.
    """

    def __init__(self, c):
        """
        Initialise the environment
        :param c: The configuration
        """
        self.c = c
        self.storage = ObjectStorage(c, self)

    def generate_objects(self, object_class, oracle):
        """
        Generate the objects with their latent representations and waypoints
        :param object_class: The class of the object
        """
        self.storage.generate_objects(object_class)
        self.storage.generate_helper_data(oracle)

    def generate_objects_ail(self, object_class):
        self.storage.generate_objects(object_class)
        self.storage.generate_helper_data_ail()
        return self.get_obj_to_similarity_list_dict()

    def try_trajectory(self, trajectory, waypoints):
        """
        Try a trajectory to see if it reaches the waypoints
        :param trajectory: The trajectory to try
        :param waypoints: The waypoints to reach
        :return: True if the trajectory reaches the waypoints and False otherwise
        """
        current_position = np.zeros_like(waypoints[0])
        waypoint_index = 0
        for action in trajectory:
            current_position += action
            if np.allclose(
                current_position,
                waypoints[waypoint_index],
                atol=self.c.POSITION_TOLERANCE,
            ):
                waypoint_index += 1
                if waypoint_index == len(waypoints):
                    break
        return waypoint_index == len(waypoints)

    def generate_trajectory(self, waypoints):
        """
        Generate a trajectory that goes through the waypoints
        :param waypoints: The waypoints to go through
        :return: The trajectory
        """
        trajectory = []
        trajectory.extend(
            self.generate_trajectory_between(np.zeros_like(waypoints[0]), waypoints[0])
        )
        for i in range(len(waypoints) - 1):
            trajectory.extend(
                self.generate_trajectory_between(waypoints[i], waypoints[i + 1])
            )
        trajectory = np.array(trajectory)
        assert self.try_trajectory(trajectory, waypoints)
        return trajectory

    def generate_trajectory_between(self, start_pos, end_pos):
        """
        Generate a trajectory between two points
        :param start_pos: The start position
        :param end_pos: The end position
        :return: The trajectory
        """
        num_steps = max(
            self.c.MIN_TRAJ_STEPS,
            np.ceil(np.linalg.norm(end_pos - start_pos) / self.c.MAX_ACTION * 2).astype(
                int
            ),
        )
        traj = np.linspace(start_pos, end_pos, num_steps)
        actions = np.diff(traj, axis=0)
        for i in range(actions.shape[0] - 1):
            noise = np.random.normal(
                0, self.c.ACTION_EXPLORATION_DEVIATION, actions[i].shape
            )
            actions[i] += noise
            actions[i + 1] -= noise
        assert np.allclose(
            np.sum(actions, axis=0) + start_pos, end_pos, atol=self.c.POSITION_TOLERANCE
        )
        return actions

    def select_object_to_try(self):
        """
        Select an object to try
        :return: The index of the object to try
        """
        return self.storage.select_object_to_try()

    def update_object(self, obj, success, trajectory):
        """
        Update the object with the success of the trajectory
        :param obj: The object to update
        :param success: Whether the trajectory was successful
        :param trajectory: The trajectory that was tried
        """
        self.storage.update_object(obj, success, trajectory)

    def has_unknown_objects(self):
        """
        Check if there are unknown objects
        :return: True if there are unknown objects and False otherwise
        """
        return self.storage.has_unknown_objects()

    def find_most_similar_known_object_demo(self, u):
        """
        Find the most similar known object to the unknown object
        :param u: The unknown object index
        :return: The demonstration of the most similar known object
        """
        return self.storage.find_most_similar_known_object_demo(u)

    def find_top_k_most_similar_known_objects_demo(self, u):
        """
        Find the top k most similar known objects to the unknown object
        :param u: The unknown object index
        :return: The demonstrations of the top k most similar known objects
        """
        return self.storage.find_top_k_most_similar_known_objects_demo(u)

    def get_objects(self):
        """
        Get all the objects in the environment
        :return: The objects
        """
        return self.storage.objects

    def get_position(self, obj_ind):
        """
        Get the position of an object
        :param obj_ind: The index of the object
        :return: The position of the object
        """
        return self.storage.objects[obj_ind].get_position()

    def try_object_trajectory(self, obj_ind, trajectory):
        """
        Try a trajectory for an object
        :param obj_ind: The index of the object
        :param trajectory: The trajectory to try
        """
        return self.try_trajectory(trajectory, self.storage.objects[obj_ind].waypoints)

    @staticmethod
    def get_trajectory_end(trajectory):
        """
        Get the end position of a trajectory
        :param trajectory: The trajectory
        :return: The end position
        """
        return np.sum(trajectory, axis=0)

    def get_selection_frequency(self):
        """
        Get the selection frequency of the known objects
        :return: The selection frequency of the known objects
        """
        return self.storage.selection_frequency

    def get_similarities(self):
        return self.storage.visual_similarities_by_task

    def get_latent_similarity(self, o, s):
        return self.storage.get_latent_similarity(o.index, s.index)

    def get_visual_similarity(self, o, s):
        return self.storage.get_visual_similarity(o.index, s.index)

    def try_transfer(self, obj, other):
        return (
            self.storage.get_latent_similarity(obj.index, other.index)
            > self.c.SIMILARITY_THRESHOLD
        ) and obj.task_type == other.task_type

    def get_obj_to_similarity_list_dict(self):
        similarity_dict = {}
        similarities = self.get_similarities()
        for o in self.get_objects():
            s = similarities[o.task_type][o.index]
            ar = []
            for o2 in self.get_objects():
                if o.task_type != o2.task_type:
                    continue
                ar.append((o2.index, s[o2.index]))
            ss = sorted(ar, key=lambda x: x[1])
            similarity_dict[o.index] = ([x[0] for x in ss], [x[1] for x in ss])
        return similarity_dict
