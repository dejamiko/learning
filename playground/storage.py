import json
import os

import numpy as np
from sklearn.cluster import KMeans

from playground.basic_object import BasicObject
from playground.sim_object import SimObject
from playground.task_types import Task
from utils import SingletonMeta, set_seed


class ObjectStorage(metaclass=SingletonMeta):
    """
    A singleton class to store objects, either randomly generated or loaded from the DINOBot data.
    """

    def __init__(self, c):
        """
        Initialize the object storage
        :param c: the configuration object
        """
        self._set_fields(c)

    def reset(self, c):
        """
        Reset the storage
        :param c: the configuration object
        """
        self._set_fields(c)

    def generate_objects(self):
        """
        Populate the storage with objects - either randomly generated or loaded from DINOBot data.
        """
        if self.c.USE_REAL_OBJECTS:
            self._ingest_data()
            self._generate_visual_similarities()
        else:
            self._generate_random_objects()
            self._generate_visual_similarities()

    def get_visual_similarity(self, i, j):
        """
        Get the visual similarity for two objects by their indices.
        :param i: the first object
        :param j: the second object
        :return: The visual similarity value between 0.0 and 1.0
        """
        return self._visual_similarities_by_task[self._objects[i].task][i, j]

    def get_true_success_probability(self, i, j, sim_threshold):
        """
        Get the true probability of success of transfer for two objects by their indices. Depending on the config,
        return a boolean value for success or a float probability.
        :param i: the first object
        :param j: the second object
        :param sim_threshold: The similarity threshold used to determine success
        :return: The real success probability value between 0.0 and 1.0
        """
        if self.c.SUCCESS_RATE_BOOLEAN:
            if self._latent_similarities[i, j] >= sim_threshold:
                return 1.0
            return 0.0

        return self._latent_similarities[i, j]

    def get_estimated_success_probability(self, i, j, sim_threshold):
        """
        Get an estimated probability of success of transfer for two objects by their indices. Depending on the config,
        return a boolean value for success or a float probability.
        :param i: the first object
        :param j: the second object
        :param sim_threshold: The similarity threshold used to determine success
        :return: The estimated success probability value between 0.0 and 1.0
        """
        if self.c.SUCCESS_RATE_BOOLEAN:
            if self.get_visual_similarity(i, j) > sim_threshold:
                return 1.0
            return 0.0

        # TODO how to best map visual sim to prob of success? Beta distribution? Something that matches the data?
        return self._visual_similarities_by_task[self._objects[i].task][i, j]

    def get_objects(self):
        """
        Return the objects
        :return: The objects stored
        """
        return self._objects

    def _ingest_data(self):
        """
        Ingest the real data for simulation objects. This assumes the data from DINOBot is stored in the `_data`
        directory at the root of the project and contains the following:
        - `objects.json` file with the object data in the format `object_name`-`task`: `image_path`, `image_embeddings`
        - `transfers.json` file with the transfer data in the format `object_name_1`-`object_name_2`-`task`:
            `success_rate`
        - `object_name`_`task` directories for all objects with the demonstration images
        """
        with open(os.path.join("_data", "objects.json")) as f:
            object_data = json.load(f)
        with open(os.path.join("_data", "transfers.json")) as f:
            transfer_data = json.load(f)
        objects = []
        for o in object_data.keys():
            img_path = object_data[o]["image_path"]
            embeddings = object_data[o]["image_embeddings"]
            name, task = o.split("-")
            objects.append(
                SimObject(len(objects), self.c, task, name, img_path, embeddings)
            )
        self._objects = np.array(objects, dtype=SimObject)
        # important to update the config object
        self.c.OBJ_NUM = len(self._objects)

        latent_similarities = np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM))
        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                if objects[i].task != objects[j].task:
                    continue
                # the data is already within 0 and 1
                latent_similarities[i, j] = transfer_data[
                    f"{objects[i].name}-{objects[j].name}-{objects[i].task}"
                ]
        self._latent_similarities = latent_similarities

    def _generate_random_objects(self):
        """
        Generate the objects with random latent representations
        """
        objects = []
        for i in range(self.c.OBJ_NUM):
            objects.append(
                BasicObject(
                    i,
                    self.c,
                    Task.HAMMERING.value,
                    np.random.uniform(0, 10, self.c.LATENT_DIM),
                )
                # the task is a placeholder here
            )
        # cluster the objects into different task types based on the latent representation
        k = KMeans(n_clusters=len(Task)).fit(np.array([o.latent_repr for o in objects]))
        task_labels = list(t.value for t in Task)
        for i, o in enumerate(objects):
            o.task = task_labels[k.labels_[i]]

        self._objects = np.array(objects, dtype=BasicObject)

        latent_similarities = np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM))
        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                latent_similarities[i, j] = objects[i].get_latent_similarity(objects[j])

        # apply min-max normalization to the latent similarities
        min_ = latent_similarities.min()
        max_ = latent_similarities.max()
        self._latent_similarities = (latent_similarities - min_) / (max_ - min_)

        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                if objects[i].task != objects[j].task:
                    self._latent_similarities[i, j] = 0.0

    def _generate_visual_similarities(self):
        """
        Generate the visual similarity data
        """
        visual_similarities = np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM))
        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                visual_similarities[i, j] = self._objects[i].get_visual_similarity(
                    self._objects[j]
                )

        # apply min-max normalization to the visual similarities
        min_ = visual_similarities.min()
        max_ = visual_similarities.max()
        visual_similarities = (visual_similarities - min_) / (max_ - min_)

        self._visual_similarities_by_task = {
            t.value: np.zeros((self.c.OBJ_NUM, self.c.OBJ_NUM)) for t in Task
        }

        for i in range(self.c.OBJ_NUM):
            for j in range(self.c.OBJ_NUM):
                if self._objects[i].task == self._objects[j].task:
                    self._visual_similarities_by_task[self._objects[i].task][i, j] = (
                        visual_similarities[i, j]
                    )

    def _set_fields(self, c):
        """
        Set the object fields and the seed for reproducibility
        :param c: the configuration object
        """
        self.c = c
        set_seed(c.SEED)
        self._objects = None
        self._visual_similarities_by_task = None
        self._latent_similarities = None
