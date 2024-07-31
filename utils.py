from enum import Enum, auto
from threading import Lock

import numpy as np


class SingletonMeta(type):
    """
    A thread-safe singleton implementation taken from https://refactoring.guru/design-patterns/singleton/python
    """

    _instances = {}

    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


def set_seed(seed):  # pragma: no cover
    np.random.seed(seed)


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


class ThresholdEstimationStrategy(AutoName):
    DENSITY = auto()
    RANDOM = auto()
    INTERVALS = auto()
    GREEDY = auto()


class SimilarityMeasure(AutoName):
    DINO_LAYER_9_COSINE = auto()
