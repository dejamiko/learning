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
        self._generate_objects()

    def _generate_objects(self):
        self.storage.generate_objects()
        self.storage.generate_helper_data()
        self.similarity_dict = self.get_obj_to_similarity_list_dict()

    def get_objects(self):
        """
        Get all the objects in the environment
        :return: The objects
        """
        return self.storage.objects

    def get_similarities(self):
        return self.storage.visual_similarities_by_task.copy()

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

    def get_reachable_object_indices(self, selected_indices):
        reachable_indices_to_selected = {}
        for o in self.get_objects():
            reachable_indices_to_selected[o.index] = set()
            for s in selected_indices:
                if self.try_transfer(o, self.get_objects()[s]):
                    reachable_indices_to_selected[o.index].add(
                        (s, self.get_latent_similarity(o, self.get_objects()[s]))
                    )
        for o in self.get_objects():
            if len(reachable_indices_to_selected[o.index]) == 0:
                reachable_indices_to_selected.pop(o.index)
        return reachable_indices_to_selected

    def evaluate_selection_transfer_based(self, selected):
        count = 0
        selected = np.where(selected == 1)[0]
        selected = self.get_objects()[selected]
        for o in self.get_objects():
            for s in selected:
                if self.try_transfer(o, s):
                    count += 1
                    break
        return count

    def evaluate_selection_similarity_based(self, selected, threshold=None):
        if threshold is None:
            threshold = self.c.SIMILARITY_THRESHOLD
        objects = set()
        for i, s in enumerate(selected):
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
