from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff, cdist

from playground.model_training import SiameseNetwork
from tm_utils import (
    Task,
    SimilarityMeasure,
    ContourSimilarityMeasure,
    ContourImageEmbeddings,
    ImageEmbeddings,
    NNSimilarityMeasure,
)


class Object(ABC):
    """
    The basic class for objects in the environment. Each object has a latent representation, a visible representation,
    and a task type.
    """

    def __init__(self, index, c, task):
        """
        Initialise the object
        :param index: The index of the object
        :param c: The configuration object
        :param task: The task this object is prepared for
        """
        self.index = index
        self.c = c

        self.task = Task(task)
        self.visible_repr = None

    def get_visual_similarity(self, other):
        this_vis_repr = self.visible_repr
        other_vis_repr = other.visible_repr

        if (
            self.c.SIMILARITY_MEASURE in SimilarityMeasure
            and len(this_vis_repr.shape) == 1
        ):
            this_vis_repr = this_vis_repr.reshape(1, *this_vis_repr.shape)
            other_vis_repr = other_vis_repr.reshape(1, *other_vis_repr.shape)
        elif self.c.SIMILARITY_MEASURE in ContourSimilarityMeasure:
            this_vis_repr = self._ensure_3d(this_vis_repr)
            other_vis_repr = self._ensure_3d(other_vis_repr)

        aggregate = []
        for a, b in zip(this_vis_repr, other_vis_repr):
            aggregate.append(self._get_visual_similarity_for_vectors(a, b))

        return sum(aggregate) / len(aggregate)

    def _get_visual_similarity_for_vectors(self, a, b):
        match self.c.SIMILARITY_MEASURE:
            case SimilarityMeasure.COSINE:
                assert self.c.IMAGE_EMBEDDINGS in ImageEmbeddings, (
                    f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` "
                    f"do not work with non-contour similarity measures."
                )
                return self._get_cos_sim(a, b)
            case SimilarityMeasure.EUCLIDEAN:
                assert self.c.IMAGE_EMBEDDINGS in ImageEmbeddings, (
                    f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` "
                    f"do not work with non-contour similarity measures."
                )
                return self._get_euclidean(a, b)
            case SimilarityMeasure.MANHATTAN:
                assert self.c.IMAGE_EMBEDDINGS in ImageEmbeddings, (
                    f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` "
                    f"do not work with non-contour similarity measures."
                )
                return self._get_manhattan(a, b)
            case SimilarityMeasure.PEARSON:
                assert self.c.IMAGE_EMBEDDINGS in ImageEmbeddings, (
                    f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` "
                    f"do not work with non-contour similarity measures."
                )
                return self._get_pearson(a, b)
            case ContourSimilarityMeasure.HAUSDORFF:
                assert self.c.IMAGE_EMBEDDINGS in ContourImageEmbeddings, (
                    f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` "
                    f"do not work with contour similarity measures."
                )
                return self._get_hausdorff(a, b)
            case ContourSimilarityMeasure.ASD:
                assert self.c.IMAGE_EMBEDDINGS in ContourImageEmbeddings, (
                    f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` "
                    f"do not work with contour similarity measures."
                )
                return self._get_asd(a, b)
            case NNSimilarityMeasure.TRAINED:
                assert (
                    self.c.IMAGE_EMBEDDINGS == ImageEmbeddings.OWN_TRAINED
                ), f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` do not work with own models."
                return self._get_own_trained(a, b)
            case NNSimilarityMeasure.FINE_TUNED:
                assert (
                    self.c.IMAGE_EMBEDDINGS == ImageEmbeddings.OWN_TRAINED
                ), f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` do not work with own models."
                return self._get_fine_tuned(a, b)
            case NNSimilarityMeasure.LINEARLY_PROBED:
                assert (
                    self.c.IMAGE_EMBEDDINGS == ImageEmbeddings.OWN_TRAINED
                ), f"The ImageEmbeddings provided `{self.c.IMAGE_EMBEDDINGS}` do not work with own models."
                return self._get_linearly_probed(a, b)
        raise ValueError(
            f"Unknown similarity measure provided `{self.c.SIMILARITY_MEASURE}`."
        )

    def __repr__(self):
        """
        This is used for printing collections of objects in a readable way.
        """
        return self.__str__()

    @abstractmethod
    def __str__(self):
        pass  # pragma: no cover

    @staticmethod
    def _get_cos_sim(a, b, eps=1e-8):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

    @staticmethod
    def _get_euclidean(a, b):
        return 1 / (1 + np.linalg.norm(a - b))

    @staticmethod
    def _get_manhattan(a, b):
        return 1 / (1 + np.sum(np.abs(a - b)))

    @staticmethod
    def _get_pearson(a, b):
        return np.corrcoef(a, b)[0, 1]

    @staticmethod
    def _get_hausdorff(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0

        hausdorff_dist = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
        return 1 / (1 + hausdorff_dist)

    @staticmethod
    def _get_asd(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0

        dist_matrix = cdist(a, b)

        # Calculate the average minimum distance from points1 to points2
        min_distances_1_to_2 = np.min(dist_matrix, axis=1)
        avg_dist_1_to_2 = np.mean(min_distances_1_to_2)

        # Calculate the average minimum distance from points2 to points1
        min_distances_2_to_1 = np.min(dist_matrix, axis=0)
        avg_dist_2_to_1 = np.mean(min_distances_2_to_1)

        # The average surface distance is the mean of these two values
        asd = (avg_dist_1_to_2 + avg_dist_2_to_1) / 2.0
        return 1 / (1 + asd)

    @staticmethod
    def _ensure_3d(lst):
        if not isinstance(lst, list):
            return lst

        if all(
            isinstance(i, list) and all(not isinstance(j, list) for j in i) for i in lst
        ):
            return [lst]

        if all(
            isinstance(i, list)
            and all(
                isinstance(j, list) and all(not isinstance(k, list) for k in j)
                for j in i
            )
            for i in lst
        ):
            return lst

    @staticmethod
    def _get_own_trained(a, b):
        return Object._get_nn_sim(
            a, b, {"frozen": False, "backbone": False}, "siamese_network_train.pth"
        )

    @staticmethod
    def _get_fine_tuned(a, b):
        return Object._get_nn_sim(
            a, b, {"frozen": True, "backbone": True}, "siamese_network_fine_tuning.pth"
        )

    @staticmethod
    def _get_linearly_probed(a, b):
        return Object._get_nn_sim(
            a,
            b,
            {"frozen": False, "backbone": True},
            "siamese_network_linear_probing.pth",
        )

    @staticmethod
    def _get_nn_sim(a, b, config, model_path):
        a = torch.from_numpy(a).float().reshape(1, 3, 256, 256)
        b = torch.from_numpy(b).float().reshape(1, 3, 256, 256)
        model = SiameseNetwork(**config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return float(model(a, b).squeeze().detach().numpy())
