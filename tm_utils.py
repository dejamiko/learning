from enum import Enum, auto

import numpy as np


def get_rng(seed):  # pragma: no cover
    return np.random.default_rng(seed)


def get_object_indices(selected):
    return np.where(selected == 1)[0]


def get_bin_representation(selected, max_len):
    selected_z = np.zeros(max_len)
    selected_z[selected] = 1
    return selected_z


class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class Task(AutoName):
    GRASPING = auto()
    PUSHING = auto()
    HAMMERING = auto()

    @classmethod
    def get_ind(cls, task):
        if task == cls.GRASPING:
            return 0
        if task == cls.PUSHING:
            return 1
        if task == cls.HAMMERING:
            return 2


class VisualisationMethod(AutoName):
    PCA = auto()
    TSNE = auto()
    UMAP = auto()


class ObjectSelectionStrategyThreshold(AutoName):
    RANDOM = auto()
    GREEDY = auto()
    DENSITY = auto()
    INTERVALS = auto()


class ObjectSelectionStrategyAffine(AutoName):
    RANDOM = auto()
    GREEDY_P = auto()
    GREEDY_R = auto()


class SimilarityMeasure(AutoName):
    COSINE = auto()
    EUCLIDEAN = auto()
    MANHATTAN = auto()
    PEARSON = auto()


class ContourSimilarityMeasure(AutoName):
    HAUSDORFF = auto()
    ASD = auto()


class NNSimilarityMeasure(AutoName):
    TRAINED = auto()
    FINE_TUNED = auto()
    LINEARLY_PROBED = auto()


class ImageEmbeddings(AutoName):
    DINO_LAYER_9 = auto()
    DINO_LAYER_11 = auto()
    DINO_FULL = auto()
    DINO_2_FULL = auto()
    VIT = auto()
    CONVNET = auto()
    SWIN = auto()
    VIT_MSN = auto()
    CLIP = auto()
    DOBBE = auto()
    VC = auto()
    R3M = auto()


class ContourImageEmbeddings(AutoName):
    MASK_RCNN = auto()
    PANOPTIC_FPN = auto()
    CASCADE_MASK_RCNN = auto()


class NNImageEmbeddings(AutoName):
    SIAMESE = auto()


class ImagePreprocessing(AutoName):
    BACKGROUND_REM = auto()
    SEGMENTATION = auto()
    GREYSCALE = auto()
    CROPPING = auto()
