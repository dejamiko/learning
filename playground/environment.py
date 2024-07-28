import numpy as np

from playground.storage import ObjectStorage
from utils import get_object_indices, set_seed


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
        set_seed(c.SEED)
        self.storage = ObjectStorage(c)
        self._generate_objects()

    def get_objects(self):
        """
        Get all the objects in the environment
        :return: The objects in the environment
        """
        return self.storage.get_objects()

    def get_visual_similarity(self, i, j):
        """
        Get the visual similarity between two objects.
        :param i: The first object index
        :param j: The second object index
        :return: The visual similarity between objects
        """
        return self.storage.get_visual_similarity(i, j)

    def get_reachable_object_indices(self, selected_indices):
        """
        Get the reachable object indices from the provided selected_indices. Returns a dictionary: reachable_index:
        list of (selected_index, distance)
        :param selected_indices: The selected indices to be used
        :return: The above described dictionary
        """
        reachable_indices_to_selected = {}
        for o in self.get_objects():
            reachable_indices_to_selected[o.index] = set()
            for s in selected_indices:
                if self.get_transfer_success(o.index, s):
                    reachable_indices_to_selected[o.index].add(
                        (s, self.get_visual_similarity(o.index, s))
                    )
        for o in self.get_objects():
            if len(reachable_indices_to_selected[o.index]) == 0:
                reachable_indices_to_selected.pop(o.index)
        return reachable_indices_to_selected

    def get_transfer_success(self, i, j):
        """
        Get the transfer success for two object indices
        :param i: The first object index
        :param j: The second object index
        :return: True if the transfer was successful, False otherwise
        """
        return np.random.random() < self._get_real_transfer_probability(i, j)

    def evaluate_selection_transfer_based(self, selected_bin):
        """
        Evaluate the given selection based on transfer probabilities.
        :param selected_bin: The selection of objects given in the binary form
        :return: The number of learned objects.
        """
        count = 0
        selected = get_object_indices(selected_bin)
        selected_obj = self.get_objects()[selected]
        for o in self.get_objects():
            for s in selected_obj:
                if np.random.random() < self._get_real_transfer_probability(
                    o.index, s.index
                ):
                    count += 1
                    break
        return count

    def evaluate_selection_visual_similarity_based(self, selected_bin, threshold=None):
        """
        Evaluate the given selection based a more efficient sorted similarity representation.
        Note this evaluation is not "real" and is instead an estimate.
        :param selected_bin: The selection of objects given in the binary form
        :param threshold: An optional threshold parameter to be used for determining the success. If not provided, the
            config value is used instead
        :return: The number of learned objects.
        """
        if threshold is None:
            threshold = self.c.SIMILARITY_THRESHOLD
        objects = set()
        for i, s in enumerate(selected_bin):
            if s == 0:
                continue
            # assume similarities[s] returns a sorted list of similarities to all objects
            # where o is an object in objects and s is an object in selected
            # use binary search to find the first object with similarity below threshold
            objs, sim = self.similarity_dict[i]
            # sim is a sorted list of similarities, use bin search to find the first object with similarity
            # below threshold
            ind = np.searchsorted(sim, threshold)
            objects.update(objs[ind:])
        return len(objects)

    def _generate_objects(self):
        """
        Populate the environment by populating the underlying storage. Create the similarity dictionary data structure.
        """
        self.storage.generate_objects()
        self.similarity_dict = self._get_obj_to_similarity_list_dict()

    def _get_obj_to_similarity_list_dict(self):
        """
        Create an efficient data structure for evaluating selections. This Dictionary stores pairs of object index to
        visual similarities between that object and every other object of the same task in a sorted list in a decreasing
        order.
        :return: The above described dictionary
        """
        similarity_dict = {}
        for o in self.get_objects():
            s = [
                self.get_visual_similarity(o.index, oth.index)
                for oth in self.get_objects()
            ]
            ar = []
            for o2 in self.get_objects():
                if o.task != o2.task:
                    continue
                ar.append((o2.index, s[o2.index]))
            ss = sorted(ar, key=lambda x: x[1])
            similarity_dict[o.index] = ([x[0] for x in ss], [x[1] for x in ss])
        return similarity_dict

    def _get_real_transfer_probability(self, i, j):
        """
        Get the real transfer probability for two objects based on the underlying storage implementation.
        :param i: The first object index
        :param j: The second object index
        :return: The real transfer probability for two objects.
        """
        return self.storage.get_true_success_probability(
            i, j, self.c.SIMILARITY_THRESHOLD
        )
