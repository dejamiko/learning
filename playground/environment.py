import numpy as np

from playground.storage import ObjectStorage
from tm_utils import get_object_indices, get_rng


def apply_affine_fun(param, f):
    return max(min(f[0] * param + f[1], 1), 0)


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
        self.storage = ObjectStorage(c)
        self.visual_sim_threshold = self.c.SIMILARITY_THRESHOLD
        self._rng = get_rng(c.SEED)
        self.similarity_dict = None
        self.similarity_matrix = None
        self._generate_objects()

    def get_objects(self):
        """
        Get all the objects in the environment
        :return: The objects in the environment
        """
        return self.storage.get_objects()

    def get_visual_similarity(self, i, j, f=(1, 0)):
        """
        Get the visual similarity between two objects.
        :param i: The first object index
        :param j: The second object index
        :param f: An optional affine function applied to the similarity
        :return: The visual similarity between objects
        """
        return apply_affine_fun(self.storage.get_visual_similarity(i, j), f)

    def get_reachable_object_indices(self, selected_indices, f=(1, 0)):
        """
        Get the reachable object indices from the provided selected_indices. Returns a dictionary: reachable_index:
        list of (selected_index, distance)
        :param selected_indices: The selected indices to be used
        :param f: An optional affine function applied to the similarity
        :return: The above described dictionary
        """
        reachable_indices_to_selected = {}
        for o in self.get_objects():
            reachable_indices_to_selected[o.index] = set()
            for s in selected_indices:
                if self.get_transfer_success(o.index, s):
                    reachable_indices_to_selected[o.index].add(
                        (s, self.get_visual_similarity(o.index, s, f))
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
        return self._rng.random() < self.get_real_transfer_probability(i, j)

    def evaluate_selection_transfer_based(self, selected_bin):
        """
        Evaluate the given selection based on transfer probabilities. Note, this implicitly works for both boolean
        and real-valued approaches.
        :param selected_bin: The selection of objects given in the binary form
        :return: The number of learned objects.
        """
        count = 0
        selected = get_object_indices(selected_bin)
        selected_obj = self.get_objects()[selected]
        for o in self.get_objects():
            for s in selected_obj:
                if self.get_transfer_success(o.index, s.index):
                    count += 1
                    break
        return count

    def evaluate_selection_visual_similarity_based(self, selected_bin):
        """
        Evaluate the given selection based a more efficient sorted similarity representation.
        Note this evaluation is not "real" and is instead an estimate.
        :param selected_bin: The selection of objects given in the binary form
        :return: The number of learned objects.
        """
        if self.c.SUCCESS_RATE_BOOLEAN:
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
                ind = np.searchsorted(sim, self.visual_sim_threshold)
                # add all objects that have a higher similarity than the threshold
                objects.update(objs[ind:])
            return len(objects)

        # treat the similarities as probabilities and maximise the overall probability of success
        # use selected_bin as a mask to select the matrix rows
        selected_rows = self.similarity_matrix[get_object_indices(selected_bin)]
        # now calculate the 1 - matrix
        selected_rows = 1 - selected_rows
        # and the column product
        prod = selected_rows.prod(axis=0)
        return prod.sum()

    def update_visual_sim_threshold(self, threshold):
        """
        Update the visual similarity threshold with a value provided
        :param threshold: The threshold to be used
        """
        self.visual_sim_threshold = threshold

    def _generate_objects(self):
        """
        Populate the environment by populating the underlying storage. Create the similarity dictionary data structure.
        """
        self.storage.generate_objects()
        self.similarity_dict = self._get_obj_to_similarity_list_dict()
        self.similarity_matrix = self._get_visual_similarity_matrix()

    def _get_obj_to_similarity_list_dict(self, f=(1, 0)):
        """
        Create an efficient data structure for evaluating selections. This Dictionary stores pairs of object index to
        visual similarities between that object and every other object of the same task in a sorted list in a decreasing
        order.
        :param f: An optional affine function applied to the similarity
        :return: The above described dictionary
        """
        similarity_dict = {}
        for o in self.get_objects():
            s = [
                self.get_visual_similarity(o.index, oth.index, f)
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

    def _get_visual_similarity_matrix(self, f=(1, 0)):
        visual_similarity = np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM))
        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                visual_similarity[i, j] = self.get_visual_similarity(i, j, f)
        return visual_similarity

    def get_real_transfer_probability(self, i, j):
        """
        Get the real transfer probability for two objects based on the underlying storage implementation.
        Use sparingly, only when you would get the actual demo success information.
        :param i: The first object index
        :param j: The second object index
        :return: The real transfer probability for two objects.
        """
        return self.storage.get_true_success_probability(i, j, self.c.PROB_THRESHOLD)

    def update_visual_similarities(self, f):
        self.similarity_dict = self._get_obj_to_similarity_list_dict(f)
        self.similarity_matrix = self._get_visual_similarity_matrix(f)
