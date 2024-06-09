from typing import Dict

import numpy as np
from sklearn.cluster import KMeans


class ObjectStorage:
    """
    Class to store the objects and their demonstrations
    """

    def __init__(self, c, env):
        """
        Initialize the object storage
        :param c: the configuration object
        :param env: the environment object
        """
        self.c = c
        self.env = env

        self.objects: np.ndarray = np.array([])
        self.known_object_indices: np.ndarray = np.array([])
        self.unknown_object_indices: np.ndarray = np.array([])
        self.object_types: np.ndarray = np.zeros(c.OBJ_NUM)
        self.closest_known_objects: np.array = np.zeros(c.OBJ_NUM)
        self.selection_frequency: np.array = np.zeros(c.OBJ_NUM)
        self.latent_similarity: np.ndarray = np.zeros((c.OBJ_NUM, c.OBJ_NUM))

        self.visual_similarities_by_task = {
            t: np.zeros((c.OBJ_NUM, c.OBJ_NUM)) for t in c.TASK_TYPES
        }
        self.object_demos: Dict[int, np.ndarray] = {}

    def generate_objects(self, object_class):
        """
        Generate the objects with their latent representations and waypoints
        :param object_class: the class of the object
        """
        objects = []
        for i in range(self.c.OBJ_NUM):
            objects.append(
                object_class(i, np.random.uniform(-1, 1, self.c.LATENT_DIM), -1, self.c)
            )
        # cluster the objects into different task types based on the latent representation
        k = KMeans(n_clusters=len(self.c.TASK_TYPES)).fit(
            np.array([o._latent_repr for o in objects])
        )
        for i, o in enumerate(objects):
            o.task_type = self.c.TASK_TYPES[k.labels_[i]]
            o.generate_waypoints()

        self.objects = np.array(objects, dtype=object_class)

    def generate_helper_data(self, oracle):
        """
        Generate the helper data for the object storage
        """
        known_objects = np.random.choice(
            self.c.OBJ_NUM, self.c.KNOWN_OBJECT_NUM, replace=False
        )
        while len(set([self.objects[i].task_type for i in known_objects])) < len(
            self.c.TASK_TYPES
        ):
            if self.c.VERBOSITY > 0:
                print(
                    "Retrying object selection to ensure all task types are represented"
                )
            known_objects = np.random.choice(
                self.c.OBJ_NUM, self.c.KNOWN_OBJECT_NUM, replace=False
            )

        for i in known_objects:
            self.object_demos[i] = oracle.get_demo(self.env, self.objects[i], self.c)

        unknown_objects = np.array(
            [i for i in range(self.c.OBJ_NUM) if i not in known_objects]
        )

        visual_similarities = np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM))
        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                visual_similarities[i, j] = self.objects[i].get_visual_similarity(
                    self.objects[j]
                )

        # apply min-max normalization to the visual similarities
        min_ = -1
        max_ = 1
        visual_similarities = (visual_similarities - min_) / (max_ - min_)

        self.known_object_indices = known_objects
        self.unknown_object_indices = unknown_objects
        self.object_types = np.array([o.task_type for o in self.objects])

        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                if self.objects[i].task_type == self.objects[j].task_type:
                    self.visual_similarities_by_task[self.objects[i].task_type][
                        i, j
                    ] = visual_similarities[i, j]

    def generate_helper_data_ail(self):
        """
        Generate the helper data for active imitation learning specifically
        """
        # this is only the similarity matrix
        visual_similarities = np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM))
        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                visual_similarities[i, j] = self.objects[i].get_visual_similarity(
                    self.objects[j]
                )

        # apply min-max normalization to the visual similarities
        min_ = -1
        max_ = 1
        visual_similarities = (visual_similarities - min_) / (max_ - min_)

        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                self.latent_similarity[i, j] = self.objects[i].get_latent_similarity(
                    self.objects[j]
                )

        # apply min-max normalization to the latent similarities
        min_ = -1
        max_ = 1
        self.latent_similarity = (self.latent_similarity - min_) / (max_ - min_)

        self.object_types = np.array([o.task_type for o in self.objects])

        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                if self.objects[i].task_type == self.objects[j].task_type:
                    self.visual_similarities_by_task[self.objects[i].task_type][
                        i, j
                    ] = visual_similarities[i, j]

    def select_object_to_try(self):
        """
        Select the object to learn next based on the MST strategy
        :return: the index of the object to try
        """
        task_types = np.unique(self.object_types[self.known_object_indices])
        candidates = []
        for t_t in task_types:
            # find the known and unknown objects that have the task type t_t
            known_objects_t = self.known_object_indices[
                self.object_types[self.known_object_indices] == t_t
            ]
            # filter out known objects for which we do not have a demonstration
            known_objects_t = known_objects_t[
                [i in self.object_demos for i in known_objects_t]
            ]
            unknown_objects_t = self.unknown_object_indices[
                self.object_types[self.unknown_object_indices] == t_t
            ]
            if len(known_objects_t) == 0 or len(unknown_objects_t) == 0:
                continue
            # find the object from the unknown objects that is most similar to one of the known objects
            u = unknown_objects_t[
                np.argsort(
                    np.max(
                        self.visual_similarities_by_task[t_t][unknown_objects_t][
                            :, known_objects_t
                        ],
                        axis=1,
                    )
                )
            ][-1]
            # find the corresponding known object taking into account the task type
            k = known_objects_t[
                np.argmax(self.visual_similarities_by_task[t_t][u, known_objects_t])
            ]
            candidates.append((u, k))
            if self.c.VERBOSITY > 1:
                print(
                    f"Task type {t_t}, unknown object {u}, known object {k}, "
                    f"similarity {self.visual_similarities_by_task[t_t][u, k]}"
                )

        # find the pair of objects that are most similar
        u, k = candidates[
            np.argmax(
                [
                    self.visual_similarities_by_task[self.objects[u].task_type][u, k]
                    for u, k in candidates
                ]
            )
        ]
        if self.c.VERBOSITY > 1:
            print(
                f"Selected unknown object {u}, known object {k}, similarity {self.visual_similarities_by_task[self.objects[u].task_type][u, k]}"
            )
            print(
                f"Replaying demo for object {self.objects[k]} to solve object {self.objects[u]} with similarity "
                f"{self.visual_similarities_by_task[self.objects[u].task_type][u, k]}"
            )
        return u

    def update_object(self, u, success, trajectory):
        """
        Update the object u with the trajectory if the task was successful
        :param u: the index of the object
        :param success: whether the task was successful
        :param trajectory: the trajectory that was tried
        """
        if success:
            self.object_demos[u] = trajectory
        else:
            pass
        self.known_object_indices = np.append(self.known_object_indices, u)
        self.unknown_object_indices = np.array(
            [i for i in range(self.c.OBJ_NUM) if i not in self.known_object_indices]
        )

    def has_unknown_objects(self):
        """
        Check if there are any unknown objects left
        :return: True if there are unknown objects and False otherwise
        """
        return len(self.unknown_object_indices) > 0

    def find_most_similar_known_object_demo(self, u):
        """
        Find the most similar known object to the unknown object u and return its demonstration
        :param u: the index of the unknown object
        :return: the demonstration of the most similar known object
        """
        ks = self.known_object_indices[
            np.argsort(
                self.visual_similarities_by_task[self.objects[u].task_type][
                    u, self.known_object_indices
                ]
            )
        ]
        ks = ks[[i in self.object_demos for i in ks]]
        self.selection_frequency[ks[-1]] += 1
        return self.object_demos[ks[-1]]

    def find_top_k_most_similar_known_objects_demo(self, u):
        """
        Find the top k most similar known objects to the unknown object u and return their average demonstration
        :param u: the index of the unknown object
        :return: the average demonstration of the top k most similar known objects
        """
        top_k = self.known_object_indices[
            np.argsort(
                (
                    self.visual_similarities_by_task[self.objects[u].task_type][
                        u, self.known_object_indices
                    ]
                )
            )
        ]
        # find the last k that have a demo not None
        top_k = top_k[[i in self.object_demos for i in top_k]][-self.c.TOP_K :]
        self.selection_frequency[top_k] += 1
        weights = self.visual_similarities_by_task[self.objects[u].task_type][u, top_k]
        weights = weights / np.sum(weights)
        # since demonstrations can have different weights, we need to average what we can
        max_demo_len = max([len(self.object_demos[i]) for i in top_k])
        demo = np.zeros((max_demo_len, self.c.LATENT_DIM))
        demo_weights = np.zeros((max_demo_len, self.c.LATENT_DIM))
        for i in range(self.c.TOP_K):
            for j in range(len(self.object_demos[top_k[i]])):
                demo[j] += self.object_demos[top_k[i]][j]
                demo_weights[j] += weights[i] * np.ones(self.c.LATENT_DIM)
        demo = demo / demo_weights
        return demo

    def get_latent_similarity(self, i, j):
        return self.latent_similarity[i, j]

    def get_visual_similarity(self, i, j):
        return self.visual_similarities_by_task[self.objects[i].task_type][i, j]
