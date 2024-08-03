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


class VisualisationMethod(AutoName):
    PCA = auto()
    TSNE = auto()
    UMAP = auto()


class ObjectSelectionStrategy(AutoName):
    DENSITY = auto()
    RANDOM = auto()
    INTERVALS = auto()
    GREEDY = auto()


class SimilarityMeasure(AutoName):
    COSINE = auto()
    EUCLIDEAN_INV = auto()
    EUCLIDEAN_EXP = auto()
    MANHATTAN_INV = auto()
    MANHATTAN_EXP = auto()
    PEARSON = auto()


class ImageEmbeddings(AutoName):
    DINO_LAYER_9 = auto()
    DINO_LAYER_11 = auto()
    DINO_FULL = auto()
    DINO_2_FULL = auto()
    MAMBA_VISION = auto()
    VIT = auto()
    CONVNET = auto()
    SWIN = auto()
