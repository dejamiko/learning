from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from playground.random_object import RandomObject


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
        self.latent_similarity: np.ndarray = np.zeros((c.OBJ_NUM, c.OBJ_NUM))

        self.visual_similarities_by_task = {
            t: np.zeros((c.OBJ_NUM, c.OBJ_NUM)) for t in c.TASK_TYPES
        }
        self.object_demos: Dict[int, np.ndarray] = {}

    def generate_random_objects(self):
        """
        Generate the objects with random latent representations
        """
        objects = []
        for i in range(self.c.OBJ_NUM):
            objects.append(
                RandomObject(i, np.random.uniform(-1, 1, self.c.LATENT_DIM), -1, self.c)
            )
        # cluster the objects into different task types based on the latent representation
        k = KMeans(n_clusters=len(self.c.TASK_TYPES)).fit(
            np.array([o.latent_repr for o in objects])
        )
        for i, o in enumerate(objects):
            o.task_type = self.c.TASK_TYPES[k.labels_[i]]

        self.objects = np.array(objects, dtype=RandomObject)

    def generate_helper_data(self):
        """
        Generate the helper data
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

        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                if self.objects[i].task_type == self.objects[j].task_type:
                    self.visual_similarities_by_task[self.objects[i].task_type][
                        i, j
                    ] = visual_similarities[i, j]

    def get_latent_similarity(self, i, j):
        return self.latent_similarity[i, j]

    def get_visual_similarity(self, i, j):
        return self.visual_similarities_by_task[self.objects[i].task_type][i, j]
