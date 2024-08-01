import os

import cv2
import numpy as np

from playground.object import Object


class SimObject(Object):
    def __init__(self, index, c, task, name, img_path, image_embeddings):
        super().__init__(index, c, task)
        self.name = name
        self.images = self._load_images(img_path)
        self.image_embeddings = np.array(image_embeddings).squeeze(1)
        self.visible_repr = self.image_embeddings[
            0
        ]  # for now use the singe top-down image

    def get_visual_similarity(self, other):
        # switch on some similarity setting the cosine similarity of the DINO embeddings
        return self._get_cos_sim(self.visible_repr, other.visible_repr)

    def __str__(self):
        return f"{self.name}, {self.visible_repr}, {self.task.value}"

    @staticmethod
    def _load_images(img_path):
        assert os.path.exists(
            img_path
        ), f"The provided image path {img_path} does not exist"
        assert os.path.isdir(
            img_path
        ), f"The provided image path {img_path} is not a directory"

        images = []
        for im in sorted(os.listdir(img_path)):
            if not im.endswith(".png"):
                continue
            images.append(cv2.imread(os.path.join(img_path, im)))
        return images
